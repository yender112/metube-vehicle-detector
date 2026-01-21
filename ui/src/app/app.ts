import { AsyncPipe, KeyValuePipe } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { AfterViewInit, Component, ElementRef, viewChild, inject, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { faTrashAlt, faCheckCircle, faTimesCircle, faRedoAlt, faSun, faMoon, faCheck, faCircleHalfStroke, faDownload, faExternalLinkAlt, faFileImport, faClock, faTachometerAlt } from '@fortawesome/free-solid-svg-icons';
import { faGithub } from '@fortawesome/free-brands-svg-icons';
import { CookieService } from 'ngx-cookie-service';
import { DownloadsService } from './services/downloads.service';
import { Themes } from './theme';
import { Download, Status, Theme, State } from './interfaces';
import { EtaPipe, SpeedPipe, FileSizePipe } from './pipes';
import { MasterCheckboxComponent, SlaveCheckboxComponent } from './components/';

@Component({
  selector: 'app-root',
  imports: [
    FormsModule,
    KeyValuePipe,
    AsyncPipe,
    FontAwesomeModule,
    NgbModule,
    EtaPipe,
    SpeedPipe,
    FileSizePipe,
    MasterCheckboxComponent,
    SlaveCheckboxComponent,
  ],
  templateUrl: './app.html',
  styleUrl: './app.sass',
})
export class App implements AfterViewInit, OnInit {
  downloads = inject(DownloadsService);
  private cookieService = inject(CookieService);
  private http = inject(HttpClient);

  addUrl!: string;
  addInProgress = false;

  // Hardcoded default values for simplified form
  private readonly DEFAULT_QUALITY = 'best';
  private readonly DEFAULT_FORMAT = 'mp4';

  themes: Theme[] = Themes;
  activeTheme: Theme | undefined;
  batchImportModalOpen = false;
  batchImportText = '';
  batchImportStatus = '';
  importInProgress = false;
  cancelImportFlag = false;
  ytDlpOptionsUpdateTime: string | null = null;
  ytDlpVersion: string | null = null;
  metubeVersion: string | null = null;

  // Download metrics
  activeDownloads = 0;
  queuedDownloads = 0;
  completedDownloads = 0;
  failedDownloads = 0;
  totalSpeed = 0;

  readonly queueMasterCheckbox = viewChild<MasterCheckboxComponent>('queueMasterCheckboxRef');
  readonly queueDelSelected = viewChild.required<ElementRef>('queueDelSelected');
  readonly queueDownloadSelected = viewChild.required<ElementRef>('queueDownloadSelected');
  readonly doneMasterCheckbox = viewChild<MasterCheckboxComponent>('doneMasterCheckboxRef');
  readonly doneDelSelected = viewChild.required<ElementRef>('doneDelSelected');
  readonly doneClearCompleted = viewChild.required<ElementRef>('doneClearCompleted');
  readonly doneClearFailed = viewChild.required<ElementRef>('doneClearFailed');
  readonly doneRetryFailed = viewChild.required<ElementRef>('doneRetryFailed');
  readonly doneDownloadSelected = viewChild.required<ElementRef>('doneDownloadSelected');

  faTrashAlt = faTrashAlt;
  faCheckCircle = faCheckCircle;
  faTimesCircle = faTimesCircle;
  faRedoAlt = faRedoAlt;
  faSun = faSun;
  faMoon = faMoon;
  faCheck = faCheck;
  faCircleHalfStroke = faCircleHalfStroke;
  faDownload = faDownload;
  faExternalLinkAlt = faExternalLinkAlt;
  faFileImport = faFileImport;
  faGithub = faGithub;
  faClock = faClock;
  faTachometerAlt = faTachometerAlt;

  constructor() {
    this.activeTheme = this.getPreferredTheme(this.cookieService);

    // Subscribe to download updates
    this.downloads.queueChanged.subscribe(() => {
      this.updateMetrics();
    });
    this.downloads.doneChanged.subscribe(() => {
      this.updateMetrics();
    });
    // Subscribe to real-time updates
    this.downloads.updated.subscribe(() => {
      this.updateMetrics();
    });
  }

  ngOnInit() {
    this.getYtdlOptionsUpdateTime();
    this.setTheme(this.activeTheme!);

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
      if (this.activeTheme && this.activeTheme.id === 'auto') {
        this.setTheme(this.activeTheme);
      }
    });
  }

  ngAfterViewInit() {
    this.downloads.queueChanged.subscribe(() => {
      this.queueMasterCheckbox()?.selectionChanged();
    });
    this.downloads.doneChanged.subscribe(() => {
      this.doneMasterCheckbox()?.selectionChanged();
      let completed = 0, failed = 0;
      this.downloads.done.forEach(dl => {
        if (dl.status === 'finished')
          completed++;
        else if (dl.status === 'error')
          failed++;
      });
      this.doneClearCompleted().nativeElement.disabled = completed === 0;
      this.doneClearFailed().nativeElement.disabled = failed === 0;
      this.doneRetryFailed().nativeElement.disabled = failed === 0;
    });
    this.fetchVersionInfo();
  }

  // workaround to allow fetching of Map values in the order they were inserted
  asIsOrder() {
    return 1;
  }

  getYtdlOptionsUpdateTime() {
    this.downloads.ytdlOptionsChanged.subscribe({
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      next: (data: any) => {
        if (data['success']) {
          const date = new Date(data['update_time'] * 1000);
          this.ytDlpOptionsUpdateTime = date.toLocaleString();
        } else {
          alert("Error reload yt-dlp options: " + data['msg']);
        }
      }
    });
  }

  getPreferredTheme(cookieService: CookieService) {
    let theme = 'auto';
    if (cookieService.check('metube_theme')) {
      theme = cookieService.get('metube_theme');
    }

    return this.themes.find(x => x.id === theme) ?? this.themes.find(x => x.id === 'auto');
  }

  themeChanged(theme: Theme) {
    this.cookieService.set('metube_theme', theme.id, { expires: 3650 });
    this.setTheme(theme);
  }

  setTheme(theme: Theme) {
    this.activeTheme = theme;
    if (theme.id === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      document.documentElement.setAttribute('data-bs-theme', 'dark');
    } else {
      document.documentElement.setAttribute('data-bs-theme', theme.id);
    }
  }

  queueSelectionChanged(checked: number) {
    this.queueDelSelected().nativeElement.disabled = checked == 0;
    this.queueDownloadSelected().nativeElement.disabled = checked == 0;
  }

  doneSelectionChanged(checked: number) {
    this.doneDelSelected().nativeElement.disabled = checked == 0;
    this.doneDownloadSelected().nativeElement.disabled = checked == 0;
  }

  addDownload(url?: string) {
    url = url ?? this.addUrl;
    if (!url) return;

    console.debug('Downloading: url=' + url);
    this.addInProgress = true;
    this.downloads.add(
      url,
      this.DEFAULT_QUALITY,
      this.DEFAULT_FORMAT,
      '',  // folder
      '',  // customNamePrefix
      0,   // playlistItemLimit
      true,
      false, // splitByChapters
      this.downloads.configuration['OUTPUT_TEMPLATE_CHAPTER'] || ''
    ).subscribe((status: Status) => {
      if (status.status === 'error') {
        alert(`Error adding URL: ${status.msg}`);
      } else {
        this.addUrl = '';
      }
      this.addInProgress = false;
    });
  }

  downloadItemByKey(id: string) {
    this.downloads.startById([id]).subscribe();
  }

  retryDownload(key: string, download: Download) {
    this.downloads.add(
      download.url,
      download.quality,
      download.format,
      download.folder,
      download.custom_name_prefix,
      download.playlist_item_limit,
      true,
      download.split_by_chapters,
      download.chapter_template
    ).subscribe();
    this.downloads.delById('done', [key]).subscribe();
  }

  delDownload(where: State, id: string) {
    this.downloads.delById(where, [id]).subscribe();
  }

  startSelectedDownloads(where: State) {
    this.downloads.startByFilter(where, dl => !!dl.checked).subscribe();
  }

  delSelectedDownloads(where: State) {
    this.downloads.delByFilter(where, dl => !!dl.checked).subscribe();
  }

  clearCompletedDownloads() {
    this.downloads.delByFilter('done', dl => dl.status === 'finished').subscribe();
  }

  clearFailedDownloads() {
    this.downloads.delByFilter('done', dl => dl.status === 'error').subscribe();
  }

  retryFailedDownloads() {
    this.downloads.done.forEach((dl, key) => {
      if (dl.status === 'error') {
        this.retryDownload(key, dl);
      }
    });
  }

  downloadSelectedFiles() {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    this.downloads.done.forEach((dl, _) => {
      if (dl.status === 'finished' && dl.checked) {
        const link = document.createElement('a');
        link.href = this.buildDownloadLink(dl);
        link.setAttribute('download', dl.filename);
        link.setAttribute('target', '_self');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    });
  }

  buildDownloadLink(download: Download) {
    let baseDir = this.downloads.configuration["PUBLIC_HOST_URL"];
    if (download.quality == 'audio' || download.filename.endsWith('.mp3')) {
      baseDir = this.downloads.configuration["PUBLIC_HOST_AUDIO_URL"];
    }

    if (download.folder) {
      baseDir += download.folder + '/';
    }

    return baseDir + encodeURIComponent(download.filename);
  }

  buildResultItemTooltip(download: Download) {
    const parts = [];
    if (download.msg) {
      parts.push(download.msg);
    }
    if (download.error) {
      parts.push(download.error);
    }
    return parts.join(' | ');
  }

  buildChapterDownloadLink(download: Download, chapterFilename: string) {
    let baseDir = this.downloads.configuration["PUBLIC_HOST_URL"];
    if (download.quality == 'audio' || chapterFilename.endsWith('.mp3')) {
      baseDir = this.downloads.configuration["PUBLIC_HOST_AUDIO_URL"];
    }

    if (download.folder) {
      baseDir += download.folder + '/';
    }

    return baseDir + encodeURIComponent(chapterFilename);
  }

  getChapterFileName(filepath: string) {
    // Extract just the filename from the path
    const parts = filepath.split('/');
    return parts[parts.length - 1];
  }

  // Open the Batch Import modal
  openBatchImportModal(): void {
    this.batchImportModalOpen = true;
    this.batchImportText = '';
    this.batchImportStatus = '';
    this.importInProgress = false;
    this.cancelImportFlag = false;
  }

  // Close the Batch Import modal
  closeBatchImportModal(): void {
    this.batchImportModalOpen = false;
  }

  // Start importing URLs from the batch modal textarea
  startBatchImport(): void {
    const urls = this.batchImportText
      .split(/\r?\n/)
      .map(url => url.trim())
      .filter(url => url.length > 0);
    if (urls.length === 0) {
      alert('No valid URLs found.');
      return;
    }
    this.importInProgress = true;
    this.cancelImportFlag = false;
    this.batchImportStatus = `Starting to import ${urls.length} URLs...`;
    let index = 0;
    const delayBetween = 1000;
    const processNext = () => {
      if (this.cancelImportFlag) {
        this.batchImportStatus = `Import cancelled after ${index} of ${urls.length} URLs.`;
        this.importInProgress = false;
        return;
      }
      if (index >= urls.length) {
        this.batchImportStatus = `Finished importing ${urls.length} URLs.`;
        this.importInProgress = false;
        return;
      }
      const url = urls[index];
      this.batchImportStatus = `Importing URL ${index + 1} of ${urls.length}: ${url}`;
      // Use hardcoded defaults for batch import
      this.downloads.add(
        url,
        this.DEFAULT_QUALITY,
        this.DEFAULT_FORMAT,
        '',  // folder
        '',  // customNamePrefix
        0,   // playlistItemLimit
        true,
        false, // splitByChapters
        this.downloads.configuration['OUTPUT_TEMPLATE_CHAPTER'] || ''
      ).subscribe({
        next: (status: Status) => {
          if (status.status === 'error') {
            alert(`Error adding URL ${url}: ${status.msg}`);
          }
          index++;
          setTimeout(processNext, delayBetween);
        },
        error: (err: Error) => {
          console.error(`Error importing URL ${url}:`, err);
          index++;
          setTimeout(processNext, delayBetween);
        }
      });
    };
    processNext();
  }

  // Cancel the batch import process
  cancelBatchImport(): void {
    if (this.importInProgress) {
      this.cancelImportFlag = true;
      this.batchImportStatus += ' Cancelling...';
    }
  }

  fetchVersionInfo(): void {
    // eslint-disable-next-line no-useless-escape
    const baseUrl = `${window.location.origin}${window.location.pathname.replace(/\/[^\/]*$/, '/')}`;
    const versionUrl = `${baseUrl}version`;
    this.http.get<{ 'yt-dlp': string, version: string }>(versionUrl)
      .subscribe({
        next: (data) => {
          this.ytDlpVersion = data['yt-dlp'];
          this.metubeVersion = data.version;
        },
        error: () => {
          this.ytDlpVersion = null;
          this.metubeVersion = null;
        }
      });
  }

  // Retry a failed video processing job
  retryProcessing(id: string) {
    this.downloads.retryProcessing(id).subscribe({
      next: (status: Status) => {
        if (status.status === 'error') {
          alert(`Error retrying processing: ${status.msg}`);
        }
      },
      error: (err: Error) => {
        console.error('Error retrying processing:', err);
        alert('Failed to retry processing');
      }
    });
  }

  private updateMetrics() {
    this.activeDownloads = Array.from(this.downloads.queue.values()).filter(d => d.status === 'downloading' || d.status === 'preparing').length;
    this.queuedDownloads = Array.from(this.downloads.queue.values()).filter(d => d.status === 'pending').length;
    this.completedDownloads = Array.from(this.downloads.done.values()).filter(d => d.status === 'finished').length;
    this.failedDownloads = Array.from(this.downloads.done.values()).filter(d => d.status === 'error').length;

    // Calculate total speed from downloading items
    const downloadingItems = Array.from(this.downloads.queue.values())
      .filter(d => d.status === 'downloading');

    this.totalSpeed = downloadingItems.reduce((total, item) => total + (item.speed || 0), 0);
  }
}
