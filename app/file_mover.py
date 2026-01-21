"""
Module to move processed videos and shots to SMB share using smbclient.
"""

import os
import shutil
import subprocess
import logging
from pathlib import Path

log = logging.getLogger('file_mover')


class FileMover:
    """Handles moving processed videos to SMB network share via smbclient."""

    def __init__(self, config):
        self.enabled = getattr(config, 'SMB_ENABLED', False)
        self.server = getattr(config, 'SMB_SERVER', '')
        self.share = getattr(config, 'SMB_SHARE', '')
        self.path = getattr(config, 'SMB_PATH', '')
        self.username = getattr(config, 'SMB_USERNAME', '')
        self.password = getattr(config, 'SMB_PASSWORD', '')
        self.domain = getattr(config, 'SMB_DOMAIN', '')

    def is_enabled(self) -> bool:
        """Check if SMB moving is enabled and configured."""
        return (self.enabled and
                bool(self.server) and
                bool(self.share))

    def _get_auth_args(self) -> list:
        """Build authentication arguments for smbclient."""
        # If no username provided, use anonymous/guest access
        if not self.username:
            return ['-N']  # -N = no password (anonymous)

        # Build auth string: DOMAIN/user%password or user%password
        if self.domain:
            auth = f"{self.domain}/{self.username}%{self.password}"
        else:
            auth = f"{self.username}%{self.password}"

        return ['-U', auth]

    def _sanitize_filename(self, name: str) -> str:
        """Remove/replace characters not valid in filenames."""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        # Replace multiple underscores with single
        while '__' in name:
            name = name.replace('__', '_')
        return name.strip('_')[:200]

    def _run_smbclient(self, commands: list) -> tuple[bool, str]:
        """
        Execute smbclient commands.

        Args:
            commands: List of smbclient commands to execute

        Returns:
            Tuple of (success: bool, message: str)
        """
        share_path = f"//{self.server}/{self.share}"
        cmd_string = "; ".join(commands)

        cmd = [
            'smbclient',
            share_path,
            self._get_auth_args(),
            '-c', cmd_string
        ]

        log.debug(f"[SMB] Executing: smbclient {share_path} -U *** -c \"{cmd_string}\"")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                log.error(f"[SMB] smbclient failed: {result.stderr}")
                return False, result.stderr

            log.debug(f"[SMB] smbclient output: {result.stdout}")
            return True, result.stdout

        except subprocess.TimeoutExpired:
            log.error("[SMB] smbclient timed out")
            return False, "Operation timed out"
        except Exception as e:
            log.error(f"[SMB] smbclient error: {e}")
            return False, str(e)

    def move_to_smb(self, video_path: str, shots_dir: str, video_title: str) -> dict:
        """
        Copy video and shots folder to SMB share, then delete local files.

        Args:
            video_path: Full path to the original video file
            shots_dir: Full path to the shots directory
            video_title: Title for organizing in destination

        Returns:
            dict with status and details
        """
        if not self.is_enabled():
            return {'status': 'disabled', 'msg': 'SMB moving not enabled'}

        safe_title = self._sanitize_filename(video_title)
        video_path = Path(video_path)
        shots_path = Path(shots_dir)

        log.info(f"[SMB] Moving files for: {video_title}")
        log.info(f"[SMB] Video: {video_path}")
        log.info(f"[SMB] Shots: {shots_path}")

        try:
            # Build base path in SMB share
            base_path = self.path.strip('/') if self.path else ''
            remote_dir = f"{base_path}/{safe_title}" if base_path else safe_title

            # Step 1: Create directory structure
            commands = []
            if base_path:
                commands.append(f'cd "{base_path}"')
            commands.append(f'mkdir "{safe_title}"')
            commands.append(f'cd "{safe_title}"')

            success, msg = self._run_smbclient(commands)
            if not success and 'NT_STATUS_OBJECT_NAME_COLLISION' not in msg:
                # Directory might already exist, which is OK
                log.warning(f"[SMB] Directory creation warning: {msg}")

            # Step 2: Upload video file
            files_to_delete = []

            if video_path.exists():
                commands = []
                if base_path:
                    commands.append(f'cd "{base_path}"')
                commands.append(f'cd "{safe_title}"')
                commands.append(f'lcd "{video_path.parent}"')
                commands.append(f'put "{video_path.name}"')

                success, msg = self._run_smbclient(commands)
                if success:
                    log.info(f"[SMB] Video uploaded: {video_path.name}")
                    files_to_delete.append(video_path)
                else:
                    return {'status': 'error', 'msg': f'Failed to upload video: {msg}'}

                # Also upload FHD scaled version if exists
                fhd_path = video_path.parent / f"{video_path.stem}_FHD{video_path.suffix}"
                if fhd_path.exists():
                    commands = []
                    if base_path:
                        commands.append(f'cd "{base_path}"')
                    commands.append(f'cd "{safe_title}"')
                    commands.append(f'lcd "{fhd_path.parent}"')
                    commands.append(f'put "{fhd_path.name}"')

                    success, msg = self._run_smbclient(commands)
                    if success:
                        log.info(f"[SMB] FHD video uploaded: {fhd_path.name}")
                        files_to_delete.append(fhd_path)
                    else:
                        log.warning(f"[SMB] Failed to upload FHD video: {msg}")

            # Step 3: Upload shots folder
            if shots_path.exists() and shots_path.is_dir():
                shot_files = list(shots_path.glob('*.jpg'))

                if shot_files:
                    # Create shots directory and upload files
                    commands = []
                    if base_path:
                        commands.append(f'cd "{base_path}"')
                    commands.append(f'cd "{safe_title}"')
                    commands.append('mkdir shots')

                    success, msg = self._run_smbclient(commands)
                    # Ignore if already exists

                    # Upload all shot files
                    commands = []
                    if base_path:
                        commands.append(f'cd "{base_path}"')
                    commands.append(f'cd "{safe_title}"')
                    commands.append('cd shots')
                    commands.append(f'lcd "{shots_path}"')
                    commands.append('prompt off')
                    commands.append('mput *.jpg')

                    success, msg = self._run_smbclient(commands)
                    if success:
                        log.info(f"[SMB] Uploaded {len(shot_files)} shot files")
                    else:
                        return {'status': 'error', 'msg': f'Failed to upload shots: {msg}'}

            # Step 4: Delete local files after successful upload
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    log.info(f"[SMB] Deleted local file: {file_path}")
                except Exception as e:
                    log.warning(f"[SMB] Failed to delete local file {file_path}: {e}")

            # Delete shots directory
            if shots_path.exists() and shots_path.is_dir():
                try:
                    shutil.rmtree(shots_path)
                    log.info(f"[SMB] Deleted local shots directory: {shots_path}")
                except Exception as e:
                    log.warning(f"[SMB] Failed to delete shots directory: {e}")

            return {
                'status': 'success',
                'destination': f'//{self.server}/{self.share}/{remote_dir}'
            }

        except Exception as e:
            log.error(f"[SMB] Failed to move files: {e}", exc_info=True)
            return {'status': 'error', 'msg': str(e)}
