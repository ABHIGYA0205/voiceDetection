/**
 * VoiceShield WebSocket Streaming Protocol, File Analysis, and Dashboard Types.
 */

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

export type RecordingState = 'idle' | 'requesting_permission' | 'recording' | 'stopping' | 'error';

export type DetectionStatus = 'low' | 'uncertain' | 'high' | 'insufficient_evidence' | 'error';

export interface QualityMetrics {
  isClipping: boolean;
  isSilent: boolean;
  hasTimingGap: boolean;
  clippingCount: number;
  gapCount: number;
}

export interface WindowHistoryItem {
  windowIndex: number;
  windowStartSec: number;
  windowEndSec: number;
  speechSeconds: number;
  isEligible: boolean;
  calibratedScore: number | null;
  rawScore: number | null;
  status: DetectionStatus;
  reason: string;
  reasonDetails: string;
  timestamp: string;
}

export interface ServerStartAck {
  type: 'session_started';
  session_id: string;
  protocol_version: string;
  canonical_sample_rate: number;
  canonical_channels: number;
  window_seconds: number;
  hop_seconds: number;
  min_speech_seconds: number;
  max_frame_bytes: number;
  max_stream_duration_seconds: number;
  idle_timeout_seconds: number;
  is_mock_detector?: boolean;
}

export interface ServerQualityEvent {
  type: 'quality_update';
  session_id: string;
  timestamp: number;
  is_clipped: boolean;
  is_silent: boolean;
  has_timing_gap: boolean;
  clipping_count: number;
  gap_count: number;
}

export interface ServerWindowResult {
  type: 'window_result';
  session_id: string;
  window_index: number;
  window_start_sec: number;
  window_end_sec: number;
  speech_seconds: number;
  is_eligible: boolean;
  raw_score?: number | null;
  calibrated_score?: number | null;
  decision: {
    state: string;
    calibrated_score: number;
    smoothed_score: number;
    reason: string;
    reason_details: string;
    window_index: number;
    history_scores?: number[];
  };
}

export interface ServerStopAck {
  type: 'session_stopped';
  session_id: string;
  total_duration_seconds: number;
  total_windows_evaluated: number;
}

export interface ServerErrorEvent {
  type: 'error';
  error_code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface FileWindowResult {
  window_index: number;
  start_seconds: number;
  end_seconds: number;
  raw_spoof_logit: number;
  raw_bonafide_logit: number;
  raw_bonafide_score: number;
  calibrated_synthetic_likelihood: number;
  is_synthetic_suspected: boolean;
  active_speech_seconds: number;
  quality_status: {
    peak_level: number;
    rms: number;
    clipping_fraction: number;
    quality_status: string;
  };
}

export interface FileInferenceResult {
  file_path: string;
  status: string;
  error_message?: string | null;
  duration_seconds: number;
  original_sample_rate: number;
  original_channels: number;
  total_windows: number;
  mean_synthetic_likelihood: number;
  max_synthetic_likelihood: number;
  is_synthetic_suspected: boolean;
  temporal_rule_explanation: string;
  is_mock_detector: boolean;
  windows: FileWindowResult[];
  metadata: Record<string, unknown>;
  disclaimers: {
    identity_proof: string;
    fraud_probability: string;
    advisory_only: string;
    threshold_policy: string;
    compliance_statement: string;
  };
}

export interface SystemReadiness {
  status: 'ready' | 'not_ready' | 'checking' | 'error';
  is_mock_detector: boolean;
  device?: string;
  reason?: string;
}
