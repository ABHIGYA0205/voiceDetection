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
  syntheticScore: number | null;
  scoreIsCalibrated: boolean;
  calibratedScore: number | null;
  rawScore: number | null;
  smoothedScore: number | null;
  thresholdLow: number;
  thresholdHigh: number;
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
  config_hash?: string;
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
  total_windows_generated?: number;
  total_windows_evaluated?: number;
  total_windows_dropped?: number;
}

export interface ServerContinuityResetEvent {
  type: 'continuity_reset';
  session_id: string;
  reason: string;
  timestamp_delta: number;
  gap_count: number;
  continuity_segment_id: number;
  timestamp: number;
}

export interface ServerWindowResult {
  type: 'window_result';
  session_id: string;
  window_index: number;
  window_start_sec: number;
  window_end_sec: number;
  speech_seconds: number;
  is_eligible: boolean;
  synthetic_score?: number | null;
  uncalibrated_synthetic_score?: number | null;
  score_is_calibrated?: boolean;
  operating_threshold?: number;
  threshold_low?: number;
  threshold_high?: number;
  operating_threshold_source?: string;
  raw_score?: number | null;
  calibrated_score?: number | null;
  decision: {
    state: string;
    synthetic_score?: number | null;
    score_is_calibrated?: boolean;
    calibrated_score?: number | null;
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
  total_windows_generated?: number;
  total_windows_dropped?: number;
  status?: string;
  final_reason?: string | null;
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
  raw_spoof_logit: number | null;
  raw_bonafide_logit: number | null;
  raw_bonafide_score: number | null;
  synthetic_score: number | null;
  uncalibrated_synthetic_score?: number | null;
  score_is_calibrated: boolean;
  calibrated_synthetic_likelihood?: number | null;
  effective_threshold?: number;
  is_synthetic_suspected: boolean;
  active_speech_seconds: number;
  is_eligible?: boolean;
  is_partial?: boolean;
  abstain_reason?: string | null;
  decision_state?: DetectionStatus | string | null;
  decision_reason?: string | null;
  decision_reason_details?: string | null;
  smoothed_score?: number | null;
  history_scores?: number[];
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
  abstain_reason?: string | null;
  duration_seconds: number;
  active_speech_duration_seconds?: number;
  original_sample_rate: number;
  original_channels: number;
  total_windows: number;
  eligible_windows?: number;
  raw_high_window_count?: number;
  uncertain_window_count?: number;
  high_window_count?: number;
  mean_synthetic_score: number | null;
  max_synthetic_score: number | null;
  mean_synthetic_likelihood?: number | null;
  max_synthetic_likelihood?: number | null;
  score_is_calibrated: boolean;
  effective_threshold: number;
  operating_threshold_source: string;
  is_synthetic_suspected: boolean;
  final_decision_state?: DetectionStatus;
  temporal_rule_explanation: string;
  is_mock_detector: boolean;
  config_hash?: string;
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
  config_hash?: string;
  reason?: string;
}
