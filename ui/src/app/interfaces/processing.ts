export interface ProcessingInfo {
    id: string;
    video_path: string;
    title: string;
    url: string;
    filename: string;
    status: 'pending' | 'scaling' | 'extracting' | 'filtering' | 'saving' | 'moving' | 'completed' | 'error';
    percent: number;
    current_step: string;
    error?: string;
    timestamp: number;
    vehicles_detected: number;
    vehicles_with_plates: number;
    shots_saved: number;
    download_dir: string;
}
