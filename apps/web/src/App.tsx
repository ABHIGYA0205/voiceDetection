import React, { useCallback, useEffect, useRef, useState } from 'react';
import './App.css';
import {
  AudioQualityMetrics,
  AudioQualityStatus,
  ConnectionState,
  DecisionState,
  DetectionStatus,
  FileInferenceResult,
  FileInferenceStatus,
  InputReliabilityAssessment,
  QualityMetrics,
  ReasonCode,
  RecordingState,
  ReliabilityReason,
  ReliabilityState,
  ServerQualityEvent,
  ServerStartAck,
  ServerStopAck,
  ServerWindowResult,
  StopStatus,
  SystemReadiness,
  TemporalDecision,
  WindowHistoryItem,
} from './types';

export function hasKey<K extends string>(obj: unknown, key: K): obj is Record<K, unknown> {
  return typeof obj === 'object' && obj !== null && Object.prototype.hasOwnProperty.call(obj, key);
}

export const REASON_ORDER: readonly ReliabilityReason[] = [
  'empty_audio',
  'silent_audio',
  'insufficient_speech',
  'severe_clipping',
  'low_energy',
  'marginal_snr',
] as const;

export const INSUFFICIENT_REASONS: ReadonlySet<ReliabilityReason> = new Set([
  'empty_audio',
  'silent_audio',
  'insufficient_speech',
]);

export const LIMITED_REASONS: ReadonlySet<ReliabilityReason> = new Set([
  'severe_clipping',
  'low_energy',
  'marginal_snr',
]);

export const ALLOWED_QUALITY_STATUSES: ReadonlySet<AudioQualityStatus> = new Set([
  'empty',
  'silent',
  'clipped',
  'low_energy',
  'marginal_snr',
  'clean',
]);

export const ALLOWED_TEMPORAL_REASON_CODES: ReadonlySet<ReasonCode> = new Set([
  'insufficient_speech_duration',
  'sustained_anomaly_elevation',
  'elevation_in_progress',
  'ambiguous_score_zone',
  'hysteresis_cooldown',
  'low_anomaly_cleared',
  'normal_acoustics',
  'signal_quality_degraded',
  'initializing',
  'insufficient_contiguous_audio',
]);

export interface AggregateCounters {
  reliable_window_count: number;
  limited_reliability_window_count: number;
  insufficient_reliability_window_count: number;
  degraded_quality_window_count: number;
  quality_origin_verdict_allowed: boolean;
}

export function validateInputReliability(raw: unknown):
  | { valid: true; value: InputReliabilityAssessment }
  | { valid: false; error: string } {
  if (!raw || typeof raw !== 'object') {
    return { valid: false, error: 'input_reliability must be a non-null object' };
  }
  const obj = raw as Record<string, unknown>;

  // explanation must be a nonempty string after trimming
  if (typeof obj.explanation !== 'string' || obj.explanation.trim().length === 0) {
    return { valid: false, error: 'explanation must contain non-whitespace text' };
  }

  // origin_verdict_allowed must be a boolean
  if (typeof obj.origin_verdict_allowed !== 'boolean') {
    return { valid: false, error: 'origin_verdict_allowed must be a boolean' };
  }

  // state must be one of reliable, limited, insufficient
  if (
    obj.state !== 'reliable' &&
    obj.state !== 'limited' &&
    obj.state !== 'insufficient'
  ) {
    return { valid: false, error: `Invalid reliability state: ${String(obj.state)}` };
  }

  // reasons must be an array
  if (!Array.isArray(obj.reasons)) {
    return { valid: false, error: 'reasons must be an array' };
  }

  const reasons = obj.reasons as ReliabilityReason[];

  // reasons must contain only recognized values
  for (const r of reasons) {
    if (!REASON_ORDER.includes(r)) {
      return { valid: false, error: `Unrecognized reliability reason: ${String(r)}` };
    }
  }

  // reasons must be unique
  if (new Set(reasons).size !== reasons.length) {
    return { valid: false, error: 'reasons cannot contain duplicate entries' };
  }

  // reasons must follow exact deterministic order
  const expectedOrdered = REASON_ORDER.filter((r) => reasons.includes(r));
  if (
    reasons.length !== expectedOrdered.length ||
    !reasons.every((r, idx) => r === expectedOrdered[idx])
  ) {
    return {
      valid: false,
      error: `reasons must appear in exact REASON_ORDER order: expected [${expectedOrdered.join(', ')}], got [${reasons.join(', ')}]`,
    };
  }

  // State-specific invariants:
  if (obj.state === 'reliable') {
    if (obj.origin_verdict_allowed !== true) {
      return { valid: false, error: 'origin_verdict_allowed must be true for state reliable' };
    }
    if (reasons.length !== 0) {
      return { valid: false, error: 'state reliable must have zero reasons' };
    }
  } else if (obj.state === 'limited') {
    if (obj.origin_verdict_allowed !== false) {
      return { valid: false, error: 'origin_verdict_allowed must be false for state limited' };
    }
    if (reasons.some((r) => INSUFFICIENT_REASONS.has(r))) {
      return { valid: false, error: 'state limited cannot contain any INSUFFICIENT_REASONS member' };
    }
    if (!reasons.some((r) => LIMITED_REASONS.has(r))) {
      return { valid: false, error: 'state limited must have at least one LIMITED_REASONS member' };
    }
  } else if (obj.state === 'insufficient') {
    if (obj.origin_verdict_allowed !== false) {
      return { valid: false, error: 'origin_verdict_allowed must be false for state insufficient' };
    }
    if (!reasons.some((r) => INSUFFICIENT_REASONS.has(r))) {
      return { valid: false, error: 'state insufficient must have at least one INSUFFICIENT_REASONS member' };
    }
  }

  return {
    valid: true,
    value: {
      state: obj.state as ReliabilityState,
      reasons,
      explanation: obj.explanation,
      origin_verdict_allowed: obj.origin_verdict_allowed,
    },
  };
}

export function validateQualityMetrics(raw: unknown):
  | { valid: true; value: AudioQualityMetrics }
  | { valid: false; error: string } {
  if (!raw || typeof raw !== 'object') {
    return { valid: false, error: 'quality_status must be a non-null object' };
  }
  const q = raw as Record<string, unknown>;

  for (const k of [
    'peak_level',
    'rms',
    'clipping_fraction',
    'is_clipped',
    'is_silent',
    'snr_db_estimate',
    'quality_status',
  ] as const) {
    if (!hasKey(q, k)) {
      return { valid: false, error: `quality_status missing required key: ${k}` };
    }
  }

  if (typeof q.peak_level !== 'number' || !Number.isFinite(q.peak_level) || q.peak_level < 0) {
    return { valid: false, error: 'peak_level must be a finite non-negative number' };
  }
  if (typeof q.rms !== 'number' || !Number.isFinite(q.rms) || q.rms < 0) {
    return { valid: false, error: 'rms must be a finite non-negative number' };
  }
  if (
    typeof q.clipping_fraction !== 'number' ||
    !Number.isFinite(q.clipping_fraction) ||
    q.clipping_fraction < 0 ||
    q.clipping_fraction > 1.0
  ) {
    return { valid: false, error: 'clipping_fraction must be a finite number in [0, 1]' };
  }
  if (typeof q.is_clipped !== 'boolean') {
    return { valid: false, error: 'is_clipped must be a boolean' };
  }
  if (typeof q.is_silent !== 'boolean') {
    return { valid: false, error: 'is_silent must be a boolean' };
  }
  if (typeof q.snr_db_estimate !== 'number' || !Number.isFinite(q.snr_db_estimate)) {
    return { valid: false, error: 'snr_db_estimate must be a finite number' };
  }
  if (
    typeof q.quality_status !== 'string' ||
    !ALLOWED_QUALITY_STATUSES.has(q.quality_status as AudioQualityStatus)
  ) {
    return { valid: false, error: `Invalid quality_status value: ${String(q.quality_status)}` };
  }

  return {
    valid: true,
    value: {
      peak_level: q.peak_level as number,
      rms: q.rms as number,
      clipping_fraction: q.clipping_fraction as number,
      is_clipped: q.is_clipped as boolean,
      is_silent: q.is_silent as boolean,
      snr_db_estimate: q.snr_db_estimate as number,
      quality_status: q.quality_status as AudioQualityStatus,
    },
  };
}

export function validateTemporalDecision(
  raw: unknown,
  enclosingWindow?: {
    window_index: number;
    window_start_sec: number;
    window_end_sec: number;
    speech_seconds: number;
  }
): { valid: true; value: TemporalDecision } | { valid: false; error: string } {
  if (!raw || typeof raw !== 'object') {
    return { valid: false, error: 'decision must be a non-null object' };
  }
  const d = raw as Record<string, unknown>;

  const requiredDecisionKeys = [
    'state',
    'synthetic_score',
    'score_is_calibrated',
    'calibrated_score',
    'smoothed_score',
    'reason',
    'reason_details',
    'window_index',
    'window_start_sec',
    'window_end_sec',
    'speech_seconds',
    'history_scores',
  ] as const;

  for (const k of requiredDecisionKeys) {
    if (!hasKey(d, k)) {
      return { valid: false, error: `decision missing required key: ${k}` };
    }
  }

  const recognizedStates: readonly DecisionState[] = [
    'low',
    'uncertain',
    'high',
    'insufficient_evidence',
  ];
  if (
    typeof d.state !== 'string' ||
    !recognizedStates.includes(d.state.toLowerCase() as DecisionState)
  ) {
    return { valid: false, error: `decision.state invalid: ${String(d.state)}` };
  }

  if (d.synthetic_score !== null) {
    if (
      typeof d.synthetic_score !== 'number' ||
      !Number.isFinite(d.synthetic_score) ||
      d.synthetic_score < 0 ||
      d.synthetic_score > 1.0
    ) {
      return {
        valid: false,
        error: 'decision.synthetic_score must be null or finite number in [0, 1]',
      };
    }
  }

  if (typeof d.score_is_calibrated !== 'boolean') {
    return { valid: false, error: 'decision.score_is_calibrated must be a boolean' };
  }

  if (d.calibrated_score !== null) {
    if (typeof d.calibrated_score !== 'number' || !Number.isFinite(d.calibrated_score)) {
      return { valid: false, error: 'decision.calibrated_score must be null or finite number' };
    }
  }

  if (
    typeof d.smoothed_score !== 'number' ||
    !Number.isFinite(d.smoothed_score) ||
    d.smoothed_score < 0 ||
    d.smoothed_score > 1.0
  ) {
    return { valid: false, error: 'decision.smoothed_score must be finite number in [0, 1]' };
  }

  if (
    typeof d.reason !== 'string' ||
    !ALLOWED_TEMPORAL_REASON_CODES.has(d.reason as ReasonCode)
  ) {
    return { valid: false, error: `decision.reason must be a recognized ReasonCode, got ${String(d.reason)}` };
  }

  if (typeof d.reason_details !== 'string' || d.reason_details.trim().length === 0) {
    return { valid: false, error: 'decision.reason_details must be a non-empty string' };
  }

  if (
    typeof d.window_index !== 'number' ||
    !Number.isInteger(d.window_index) ||
    d.window_index < 0
  ) {
    return { valid: false, error: 'decision.window_index must be a non-negative integer' };
  }

  if (
    typeof d.window_start_sec !== 'number' ||
    !Number.isFinite(d.window_start_sec) ||
    d.window_start_sec < 0
  ) {
    return { valid: false, error: 'decision.window_start_sec must be a non-negative finite number' };
  }

  if (
    typeof d.window_end_sec !== 'number' ||
    !Number.isFinite(d.window_end_sec) ||
    d.window_end_sec < d.window_start_sec
  ) {
    return { valid: false, error: 'decision.window_end_sec must be a finite number >= window_start_sec' };
  }

  if (
    typeof d.speech_seconds !== 'number' ||
    !Number.isFinite(d.speech_seconds) ||
    d.speech_seconds < 0
  ) {
    return { valid: false, error: 'decision.speech_seconds must be a non-negative finite number' };
  }

  if (!Array.isArray(d.history_scores)) {
    return { valid: false, error: 'decision.history_scores must be an array' };
  }
  for (let idx = 0; idx < d.history_scores.length; idx++) {
    const s = d.history_scores[idx];
    if (typeof s !== 'number' || !Number.isFinite(s)) {
      return { valid: false, error: `decision.history_scores[${idx}] must be a finite number` };
    }
  }

  if (enclosingWindow) {
    if (d.window_index !== enclosingWindow.window_index) {
      return {
        valid: false,
        error: `decision.window_index (${d.window_index}) does not match enclosing window_index (${enclosingWindow.window_index})`,
      };
    }
    if (Math.abs((d.window_start_sec as number) - enclosingWindow.window_start_sec) > 1e-4) {
      return {
        valid: false,
        error: `decision.window_start_sec (${d.window_start_sec}) does not match enclosing window_start_sec (${enclosingWindow.window_start_sec})`,
      };
    }
    if (Math.abs((d.window_end_sec as number) - enclosingWindow.window_end_sec) > 1e-4) {
      return {
        valid: false,
        error: `decision.window_end_sec (${d.window_end_sec}) does not match enclosing window_end_sec (${enclosingWindow.window_end_sec})`,
      };
    }
    if (Math.abs((d.speech_seconds as number) - enclosingWindow.speech_seconds) > 1e-4) {
      return {
        valid: false,
        error: `decision.speech_seconds (${d.speech_seconds}) does not match enclosing speech_seconds (${enclosingWindow.speech_seconds})`,
      };
    }
  }

  return {
    valid: true,
    value: {
      state: d.state.toLowerCase() as DecisionState,
      synthetic_score: d.synthetic_score as number | null,
      score_is_calibrated: d.score_is_calibrated as boolean,
      calibrated_score: d.calibrated_score as number | null,
      smoothed_score: d.smoothed_score as number,
      reason: d.reason as ReasonCode,
      reason_details: d.reason_details as string,
      window_index: d.window_index as number,
      window_start_sec: d.window_start_sec as number,
      window_end_sec: d.window_end_sec as number,
      speech_seconds: d.speech_seconds as number,
      history_scores: d.history_scores as number[],
    },
  };
}

export function validateFileAggregateAndWindows(rawResult: unknown):
  | { valid: true; file: FileInferenceResult }
  | { valid: false; error: string } {
  if (!rawResult || typeof rawResult !== 'object') {
    return { valid: false, error: 'File result must be a non-null object' };
  }
  const f = rawResult as Record<string, unknown>;

  const requiredFileKeys = [
    'file_path',
    'status',
    'error_message',
    'error_code',
    'abstain_reason',
    'duration_seconds',
    'active_speech_duration_seconds',
    'original_sample_rate',
    'original_channels',
    'total_windows',
    'eligible_windows',
    'reliable_window_count',
    'limited_reliability_window_count',
    'insufficient_reliability_window_count',
    'degraded_quality_window_count',
    'quality_origin_verdict_allowed',
    'raw_high_window_count',
    'uncertain_window_count',
    'high_window_count',
    'mean_synthetic_score',
    'max_synthetic_score',
    'mean_synthetic_likelihood',
    'max_synthetic_likelihood',
    'score_is_calibrated',
    'effective_threshold',
    'operating_threshold_source',
    'is_synthetic_suspected',
    'final_decision_state',
    'temporal_rule_explanation',
    'is_mock_detector',
    'config_hash',
    'windows',
    'metadata',
    'disclaimers',
  ] as const;

  for (const k of requiredFileKeys) {
    if (!hasKey(f, k)) {
      return { valid: false, error: `File result missing required key: ${k}` };
    }
  }

  // file_path
  if (typeof f.file_path !== 'string') {
    return { valid: false, error: 'file_path must be a string' };
  }

  // status
  const validStatuses: readonly FileInferenceStatus[] = [
    'success',
    'limited_reliability',
    'insufficient_evidence',
    'error',
    'unsupported',
  ];
  if (typeof f.status !== 'string' || !validStatuses.includes(f.status as FileInferenceStatus)) {
    return { valid: false, error: `Invalid status: ${String(f.status)}` };
  }

  // error_message, error_code, abstain_reason
  if (f.error_message !== null && typeof f.error_message !== 'string') {
    return { valid: false, error: 'error_message must be string or null' };
  }
  if (f.error_code !== null && typeof f.error_code !== 'string') {
    return { valid: false, error: 'error_code must be string or null' };
  }
  if (f.abstain_reason !== null && typeof f.abstain_reason !== 'string') {
    return { valid: false, error: 'abstain_reason must be string or null' };
  }

  // duration_seconds, active_speech_duration_seconds
  if (typeof f.duration_seconds !== 'number' || !Number.isFinite(f.duration_seconds) || f.duration_seconds < 0) {
    return { valid: false, error: 'duration_seconds must be a finite non-negative number' };
  }
  if (
    typeof f.active_speech_duration_seconds !== 'number' ||
    !Number.isFinite(f.active_speech_duration_seconds) ||
    f.active_speech_duration_seconds < 0
  ) {
    return { valid: false, error: 'active_speech_duration_seconds must be a finite non-negative number' };
  }

  // original_sample_rate, original_channels
  if (typeof f.original_sample_rate !== 'number' || !Number.isInteger(f.original_sample_rate) || f.original_sample_rate < 0) {
    return { valid: false, error: 'original_sample_rate must be a non-negative integer' };
  }
  if (typeof f.original_channels !== 'number' || !Number.isInteger(f.original_channels) || f.original_channels < 0) {
    return { valid: false, error: 'original_channels must be a non-negative integer' };
  }

  // total_windows
  if (typeof f.total_windows !== 'number' || !Number.isInteger(f.total_windows) || f.total_windows < 0) {
    return { valid: false, error: 'total_windows must be a finite nonnegative integer' };
  }
  const total_windows = f.total_windows as number;

  // eligible_windows
  if (typeof f.eligible_windows !== 'number' || !Number.isInteger(f.eligible_windows) || f.eligible_windows < 0) {
    return { valid: false, error: 'eligible_windows must be a finite nonnegative integer' };
  }

  // all four counters
  for (const key of [
    'reliable_window_count',
    'limited_reliability_window_count',
    'insufficient_reliability_window_count',
    'degraded_quality_window_count',
  ] as const) {
    const val = f[key];
    if (typeof val !== 'number' || !Number.isInteger(val) || val < 0) {
      return { valid: false, error: `${key} must be a finite nonnegative integer` };
    }
  }
  const reliable = f.reliable_window_count as number;
  const limited = f.limited_reliability_window_count as number;
  const insufficient = f.insufficient_reliability_window_count as number;
  const degraded = f.degraded_quality_window_count as number;

  if (reliable + limited + insufficient !== total_windows) {
    return {
      valid: false,
      error: `Counters do not sum to total_windows: ${reliable} + ${limited} + ${insufficient} !== ${total_windows}`,
    };
  }

  if (degraded < limited) {
    return {
      valid: false,
      error: `degraded_quality_window_count (${degraded}) cannot be less than limited_reliability_window_count (${limited})`,
    };
  }

  if (degraded > total_windows) {
    return {
      valid: false,
      error: `degraded_quality_window_count (${degraded}) exceeds total_windows (${total_windows})`,
    };
  }

  if (typeof f.quality_origin_verdict_allowed !== 'boolean') {
    return { valid: false, error: 'quality_origin_verdict_allowed must be a boolean' };
  }

  // raw_high_window_count, uncertain_window_count, high_window_count
  for (const key of ['raw_high_window_count', 'uncertain_window_count', 'high_window_count'] as const) {
    const val = f[key];
    if (typeof val !== 'number' || !Number.isInteger(val) || val < 0) {
      return { valid: false, error: `${key} must be a non-negative integer` };
    }
  }

  // score fields and aliases
  for (const key of [
    'mean_synthetic_score',
    'max_synthetic_score',
    'mean_synthetic_likelihood',
    'max_synthetic_likelihood',
  ] as const) {
    const val = f[key];
    if (val !== null) {
      if (typeof val !== 'number' || !Number.isFinite(val) || val < 0 || val > 1.0) {
        return { valid: false, error: `${key} must be null or finite number in [0, 1]` };
      }
    }
  }

  // score_is_calibrated
  if (typeof f.score_is_calibrated !== 'boolean') {
    return { valid: false, error: 'score_is_calibrated must be a boolean' };
  }

  // effective_threshold
  if (typeof f.effective_threshold !== 'number' || !Number.isFinite(f.effective_threshold)) {
    return { valid: false, error: 'effective_threshold must be a finite number' };
  }

  // operating_threshold_source
  if (typeof f.operating_threshold_source !== 'string' || f.operating_threshold_source.trim().length === 0) {
    return { valid: false, error: 'operating_threshold_source must be a non-empty string' };
  }

  // is_synthetic_suspected
  if (typeof f.is_synthetic_suspected !== 'boolean') {
    return { valid: false, error: 'is_synthetic_suspected must be a boolean' };
  }

  // final_decision_state
  const recognizedDecisionStates: readonly DecisionState[] = [
    'low',
    'uncertain',
    'high',
    'insufficient_evidence',
  ];
  if (
    typeof f.final_decision_state !== 'string' ||
    !recognizedDecisionStates.includes(f.final_decision_state.toLowerCase() as DecisionState)
  ) {
    return { valid: false, error: `Invalid final_decision_state: ${String(f.final_decision_state)}` };
  }

  // temporal_rule_explanation
  if (typeof f.temporal_rule_explanation !== 'string') {
    return { valid: false, error: 'temporal_rule_explanation must be a string' };
  }

  // is_mock_detector
  if (typeof f.is_mock_detector !== 'boolean') {
    return { valid: false, error: 'is_mock_detector must be a boolean' };
  }

  // config_hash
  if (f.config_hash !== null && typeof f.config_hash !== 'string') {
    return { valid: false, error: 'config_hash must be string or null' };
  }

  // metadata
  if (!f.metadata || typeof f.metadata !== 'object') {
    return { valid: false, error: 'metadata must be an object' };
  }

  // disclaimers
  if (!f.disclaimers || typeof f.disclaimers !== 'object') {
    return { valid: false, error: 'disclaimers must be an object' };
  }
  const disc = f.disclaimers as Record<string, unknown>;
  for (const dKey of ['identity_proof', 'fraud_probability', 'advisory_action', 'capture_boundary'] as const) {
    if (!hasKey(disc, dKey) || typeof disc[dKey] !== 'string') {
      return { valid: false, error: `disclaimers.${dKey} must be a string` };
    }
  }

  // Status coherence:
  if (degraded > 0 && f.quality_origin_verdict_allowed === true) {
    return {
      valid: false,
      error: 'quality_origin_verdict_allowed cannot be true when degraded_quality_window_count > 0',
    };
  }

  if (f.status === 'success') {
    if (f.quality_origin_verdict_allowed !== true) {
      return {
        valid: false,
        error: 'status success with quality_origin_verdict_allowed=false is an invalid contradictory state',
      };
    }
    if (limited > 0 || degraded > 0) {
      return {
        valid: false,
        error: 'status success with quality_origin_verdict_allowed=true cannot have limited or degraded windows',
      };
    }
    if (reliable <= 0) {
      return {
        valid: false,
        error: 'status success requires at least one reliable window',
      };
    }
    const dec = (f.final_decision_state as string).toLowerCase();
    if (dec !== 'low' && dec !== 'uncertain' && dec !== 'high') {
      return {
        valid: false,
        error: `status success requires final_decision_state in low/uncertain/high, got ${dec}`,
      };
    }
  }

  if (f.status === 'limited_reliability') {
    if (f.quality_origin_verdict_allowed === true) {
      return {
        valid: false,
        error: 'status limited_reliability cannot have quality_origin_verdict_allowed=true',
      };
    }
    if ((f.final_decision_state as string).toLowerCase() !== 'uncertain') {
      return {
        valid: false,
        error: 'status limited_reliability requires final_decision_state=uncertain',
      };
    }
  }

  if (f.status === 'insufficient_evidence') {
    if (f.quality_origin_verdict_allowed === true) {
      return {
        valid: false,
        error: 'status insufficient_evidence cannot have quality_origin_verdict_allowed=true',
      };
    }
  }

  if (f.status === 'error' || f.status === 'unsupported') {
    if (f.quality_origin_verdict_allowed === true) {
      return {
        valid: false,
        error: `status ${f.status} cannot have quality_origin_verdict_allowed=true`,
      };
    }
  }

  // eligible_windows must equal reliable_window_count
  if (f.eligible_windows !== reliable) {
    return {
      valid: false,
      error: `eligible_windows (${String(f.eligible_windows)}) does not equal reliable_window_count (${reliable})`,
    };
  }

  // windows must be an array
  if (!Array.isArray(f.windows)) {
    return { valid: false, error: 'windows must be an array' };
  }
  const windows = f.windows as Record<string, unknown>[];

  if (windows.length !== total_windows) {
    return {
      valid: false,
      error: `windows array length (${windows.length}) does not match total_windows (${total_windows})`,
    };
  }

  let derivedReliable = 0;
  let derivedLimited = 0;
  let derivedInsufficient = 0;
  let derivedDegraded = 0;

  const requiredWindowKeys = [
    'window_index',
    'start_seconds',
    'end_seconds',
    'raw_spoof_logit',
    'raw_bonafide_logit',
    'raw_bonafide_score',
    'synthetic_score',
    'uncalibrated_synthetic_score',
    'score_is_calibrated',
    'calibrated_synthetic_likelihood',
    'synthetic_likelihood',
    'effective_threshold',
    'is_synthetic_suspected',
    'active_speech_seconds',
    'is_eligible',
    'is_partial',
    'abstain_reason',
    'quality_status',
    'input_reliability',
    'decision',
    'decision_state',
    'decision_reason',
    'decision_reason_details',
    'smoothed_score',
    'history_scores',
  ] as const;

  for (let i = 0; i < windows.length; i++) {
    const win = windows[i];
    if (!win || typeof win !== 'object') {
      return { valid: false, error: `Window #${i} is not a valid object` };
    }

    for (const k of requiredWindowKeys) {
      if (!hasKey(win, k)) {
        return { valid: false, error: `Window #${i} missing required key: ${k}` };
      }
    }

    if (typeof win.window_index !== 'number' || !Number.isInteger(win.window_index) || win.window_index < 0) {
      return { valid: false, error: `Window #${i} has invalid window_index` };
    }
    if (typeof win.start_seconds !== 'number' || !Number.isFinite(win.start_seconds) || win.start_seconds < 0) {
      return { valid: false, error: `Window #${i} has invalid start_seconds` };
    }
    if (typeof win.end_seconds !== 'number' || !Number.isFinite(win.end_seconds) || win.end_seconds < win.start_seconds) {
      return { valid: false, error: `Window #${i} has invalid end_seconds` };
    }
    if (win.raw_spoof_logit !== null && (typeof win.raw_spoof_logit !== 'number' || !Number.isFinite(win.raw_spoof_logit))) {
      return { valid: false, error: `Window #${i} raw_spoof_logit must be null or finite number` };
    }
    if (win.raw_bonafide_logit !== null && (typeof win.raw_bonafide_logit !== 'number' || !Number.isFinite(win.raw_bonafide_logit))) {
      return { valid: false, error: `Window #${i} raw_bonafide_logit must be null or finite number` };
    }
    if (win.raw_bonafide_score !== null && (typeof win.raw_bonafide_score !== 'number' || !Number.isFinite(win.raw_bonafide_score))) {
      return { valid: false, error: `Window #${i} raw_bonafide_score must be null or finite number` };
    }
    if (
      win.synthetic_score !== null &&
      (typeof win.synthetic_score !== 'number' ||
        !Number.isFinite(win.synthetic_score) ||
        win.synthetic_score < 0 ||
        win.synthetic_score > 1.0)
    ) {
      return { valid: false, error: `Window #${i} synthetic_score must be null or finite number in [0, 1]` };
    }
    if (
      win.uncalibrated_synthetic_score !== null &&
      (typeof win.uncalibrated_synthetic_score !== 'number' ||
        !Number.isFinite(win.uncalibrated_synthetic_score) ||
        win.uncalibrated_synthetic_score < 0 ||
        win.uncalibrated_synthetic_score > 1.0)
    ) {
      return { valid: false, error: `Window #${i} uncalibrated_synthetic_score must be null or finite number in [0, 1]` };
    }
    if (typeof win.score_is_calibrated !== 'boolean') {
      return { valid: false, error: `Window #${i} score_is_calibrated must be a boolean` };
    }
    if (win.calibrated_synthetic_likelihood !== null && (typeof win.calibrated_synthetic_likelihood !== 'number' || !Number.isFinite(win.calibrated_synthetic_likelihood))) {
      return { valid: false, error: `Window #${i} calibrated_synthetic_likelihood must be null or finite number` };
    }
    if (
      win.synthetic_likelihood !== null &&
      (typeof win.synthetic_likelihood !== 'number' ||
        !Number.isFinite(win.synthetic_likelihood) ||
        win.synthetic_likelihood < 0 ||
        win.synthetic_likelihood > 1.0)
    ) {
      return { valid: false, error: `Window #${i} synthetic_likelihood must be null or finite number in [0, 1]` };
    }
    if (typeof win.effective_threshold !== 'number' || !Number.isFinite(win.effective_threshold)) {
      return { valid: false, error: `Window #${i} has invalid effective_threshold` };
    }
    if (typeof win.is_synthetic_suspected !== 'boolean') {
      return { valid: false, error: `Window #${i} has invalid is_synthetic_suspected` };
    }
    if (typeof win.active_speech_seconds !== 'number' || !Number.isFinite(win.active_speech_seconds) || win.active_speech_seconds < 0) {
      return { valid: false, error: `Window #${i} has invalid active_speech_seconds` };
    }
    if (typeof win.is_eligible !== 'boolean') {
      return { valid: false, error: `Window #${i} has invalid is_eligible` };
    }
    if (typeof win.is_partial !== 'boolean') {
      return { valid: false, error: `Window #${i} has invalid is_partial` };
    }
    if (win.abstain_reason !== null && typeof win.abstain_reason !== 'string') {
      return { valid: false, error: `Window #${i} abstain_reason must be string or null` };
    }

    if (win.decision_state !== null) {
      if (typeof win.decision_state !== 'string' || !recognizedDecisionStates.includes(win.decision_state.toLowerCase() as DecisionState)) {
        return { valid: false, error: `Window #${i} decision_state must be null or valid DecisionState` };
      }
    }
    if (
      win.decision_reason !== null &&
      (typeof win.decision_reason !== 'string' ||
        !ALLOWED_TEMPORAL_REASON_CODES.has(win.decision_reason as ReasonCode))
    ) {
      return { valid: false, error: `Window #${i} decision_reason must be null or a recognized ReasonCode` };
    }
    if (win.decision_reason_details !== null && typeof win.decision_reason_details !== 'string') {
      return { valid: false, error: `Window #${i} decision_reason_details must be string or null` };
    }
    if (win.smoothed_score !== null && (typeof win.smoothed_score !== 'number' || !Number.isFinite(win.smoothed_score) || win.smoothed_score < 0 || win.smoothed_score > 1.0)) {
      return { valid: false, error: `Window #${i} smoothed_score must be null or finite number in [0, 1]` };
    }

    if (!Array.isArray(win.history_scores)) {
      return { valid: false, error: `Window #${i} history_scores must be an array` };
    }
    for (let hIdx = 0; hIdx < win.history_scores.length; hIdx++) {
      const s = win.history_scores[hIdx];
      if (typeof s !== 'number' || !Number.isFinite(s)) {
        return { valid: false, error: `Window #${i} history_scores[${hIdx}] must be a finite number` };
      }
    }

    // quality_status validation
    const qVal = validateQualityMetrics(win.quality_status);
    if (!qVal.valid) {
      return { valid: false, error: `Window #${i} has invalid quality_status: ${qVal.error}` };
    }

    // input_reliability validation
    const relVal = validateInputReliability(win.input_reliability);
    if (!relVal.valid) {
      return { valid: false, error: `Window #${i} has invalid input_reliability: ${relVal.error}` };
    }
    const rel = relVal.value;

    if (rel.state === 'reliable') derivedReliable++;
    else if (rel.state === 'limited') derivedLimited++;
    else if (rel.state === 'insufficient') derivedInsufficient++;

    if (rel.reasons.some((r) => LIMITED_REASONS.has(r))) {
      derivedDegraded++;
    }

    // Decision validation when non-null
    if (win.decision !== null) {
      const decVal = validateTemporalDecision(win.decision, {
        window_index: win.window_index as number,
        window_start_sec: win.start_seconds as number,
        window_end_sec: win.end_seconds as number,
        speech_seconds: win.active_speech_seconds as number,
      });
      if (!decVal.valid) {
        return { valid: false, error: `Window #${i} has invalid decision: ${decVal.error}` };
      }
    }

    // Coherence of is_eligible with reliability state
    if (rel.state === 'reliable' && win.is_eligible !== true) {
      return { valid: false, error: `Window #${i} is reliable but is_eligible is false` };
    }
    if (rel.state !== 'reliable' && win.is_eligible !== false) {
      return { valid: false, error: `Window #${i} is non-reliable but is_eligible is true` };
    }

    // Scores for reliable windows:
    if (rel.state === 'reliable') {
      if (win.decision === null) {
        return { valid: false, error: `Window #${i} reliable window must have non-null decision` };
      }
      if (win.decision_state === null || (win.decision_state !== 'low' && win.decision_state !== 'uncertain' && win.decision_state !== 'high')) {
        return { valid: false, error: `Window #${i} reliable window decision_state must be low, uncertain, or high` };
      }
      if (typeof win.synthetic_score !== 'number' || !Number.isFinite(win.synthetic_score) || win.synthetic_score < 0 || win.synthetic_score > 1.0) {
        return { valid: false, error: `Window #${i} reliable window must have finite synthetic_score in [0, 1]` };
      }
      if (typeof win.uncalibrated_synthetic_score !== 'number' || !Number.isFinite(win.uncalibrated_synthetic_score) || win.uncalibrated_synthetic_score < 0 || win.uncalibrated_synthetic_score > 1.0) {
        return { valid: false, error: `Window #${i} reliable window must have finite uncalibrated_synthetic_score in [0, 1]` };
      }
      if (typeof win.synthetic_likelihood !== 'number' || !Number.isFinite(win.synthetic_likelihood) || win.synthetic_likelihood < 0 || win.synthetic_likelihood > 1.0) {
        return { valid: false, error: `Window #${i} reliable window must have finite synthetic_likelihood in [0, 1]` };
      }
      if (typeof win.raw_spoof_logit !== 'number' || !Number.isFinite(win.raw_spoof_logit)) {
        return { valid: false, error: `Window #${i} reliable window must have finite raw_spoof_logit` };
      }
      if (typeof win.raw_bonafide_logit !== 'number' || !Number.isFinite(win.raw_bonafide_logit)) {
        return { valid: false, error: `Window #${i} reliable window must have finite raw_bonafide_logit` };
      }
      if (typeof win.raw_bonafide_score !== 'number' || !Number.isFinite(win.raw_bonafide_score)) {
        return { valid: false, error: `Window #${i} reliable window must have finite raw_bonafide_score` };
      }
      if (typeof win.smoothed_score !== 'number' || !Number.isFinite(win.smoothed_score) || win.smoothed_score < 0 || win.smoothed_score > 1.0) {
        return { valid: false, error: `Window #${i} reliable window must have finite smoothed_score in [0, 1]` };
      }
    } else {
      // Non-reliable windows: detector scores must be explicit null
      if (win.synthetic_score !== null) {
        return { valid: false, error: `Window #${i} non-reliable window must have null synthetic_score` };
      }
      if (win.uncalibrated_synthetic_score !== null) {
        return { valid: false, error: `Window #${i} non-reliable window must have null uncalibrated_synthetic_score` };
      }
      if (win.synthetic_likelihood !== null) {
        return { valid: false, error: `Window #${i} non-reliable window must have null synthetic_likelihood` };
      }
      if (win.raw_spoof_logit !== null) {
        return { valid: false, error: `Window #${i} non-reliable window must have null raw_spoof_logit` };
      }
      if (win.raw_bonafide_logit !== null) {
        return { valid: false, error: `Window #${i} non-reliable window must have null raw_bonafide_logit` };
      }
      if (win.raw_bonafide_score !== null) {
        return { valid: false, error: `Window #${i} non-reliable window must have null raw_bonafide_score` };
      }
      // smoothed_score is temporal-policy state, not detector-origin evidence. The backend
      // emits its finite value for abstained windows, so the general nullable/finite check above applies.
      if (win.decision_state !== null && win.decision_state !== 'uncertain' && win.decision_state !== 'insufficient_evidence') {
        return { valid: false, error: `Window #${i} non-reliable window decision_state cannot be low or high` };
      }
      if (win.decision !== null) {
        const dObj = win.decision as TemporalDecision;
        if (dObj.synthetic_score !== null) {
          return { valid: false, error: `Window #${i} non-reliable window decision.synthetic_score must be null` };
        }
        if (dObj.calibrated_score !== null) {
          return { valid: false, error: `Window #${i} non-reliable window decision.calibrated_score must be null` };
        }
      }
    }
  }

  // Derived counts must equal aggregate
  if (derivedReliable !== reliable) {
    return { valid: false, error: `Derived reliable windows (${derivedReliable}) !== aggregate (${reliable})` };
  }
  if (derivedLimited !== limited) {
    return { valid: false, error: `Derived limited windows (${derivedLimited}) !== aggregate (${limited})` };
  }
  if (derivedInsufficient !== insufficient) {
    return { valid: false, error: `Derived insufficient windows (${derivedInsufficient}) !== aggregate (${insufficient})` };
  }
  if (derivedDegraded !== degraded) {
    return { valid: false, error: `Derived degraded-quality windows (${derivedDegraded}) !== aggregate (${degraded})` };
  }

  return { valid: true, file: rawResult as unknown as FileInferenceResult };
}

export function validateStreamingStopAck(raw: unknown):
  | { valid: true; stopAck: ServerStopAck }
  | { valid: false; error: string } {
  if (!raw || typeof raw !== 'object') {
    return { valid: false, error: 'Stop ACK must be a non-null object' };
  }
  const s = raw as Record<string, unknown>;

  const requiredStopKeys = [
    'type',
    'session_id',
    'total_duration_seconds',
    'total_windows_evaluated',
    'total_windows_generated',
    'total_windows_dropped',
    'status',
    'final_reason',
    'reliable_window_count',
    'limited_reliability_window_count',
    'insufficient_reliability_window_count',
    'degraded_quality_window_count',
    'quality_origin_verdict_allowed',
    'final_decision_state',
  ] as const;

  for (const k of requiredStopKeys) {
    if (!hasKey(s, k)) {
      return { valid: false, error: `session_stopped missing required key: ${k}` };
    }
  }

  if (s.type !== 'session_stopped') {
    return { valid: false, error: `type must be session_stopped, got ${String(s.type)}` };
  }

  if (typeof s.session_id !== 'string' || s.session_id.trim().length === 0) {
    return { valid: false, error: 'session_id must be a non-empty string' };
  }

  // status must be an exact recognized stop-status union:
  // 'success' | 'limited_reliability' | 'insufficient_evidence' | 'error'
  const validStopStatuses: readonly StopStatus[] = [
    'success',
    'limited_reliability',
    'insufficient_evidence',
    'error',
  ];
  if (!validStopStatuses.includes(s.status as StopStatus)) {
    return { valid: false, error: `Invalid stop status: ${String(s.status)}` };
  }

  // final_decision_state must be a valid DecisionState:
  // 'low' | 'uncertain' | 'high' | 'insufficient_evidence'
  const recognizedStates: readonly DecisionState[] = [
    'low',
    'uncertain',
    'high',
    'insufficient_evidence',
  ];
  if (
    typeof s.final_decision_state !== 'string' ||
    !recognizedStates.includes(s.final_decision_state.toLowerCase() as DecisionState)
  ) {
    return { valid: false, error: `Invalid final_decision_state: ${String(s.final_decision_state)}` };
  }

  if (typeof s.total_duration_seconds !== 'number' || !Number.isFinite(s.total_duration_seconds) || s.total_duration_seconds < 0) {
    return { valid: false, error: 'total_duration_seconds must be a finite non-negative number' };
  }

  // total_windows_generated, total_windows_evaluated, and total_windows_dropped
  for (const key of ['total_windows_generated', 'total_windows_evaluated', 'total_windows_dropped'] as const) {
    const val = s[key];
    if (typeof val !== 'number' || !Number.isInteger(val) || val < 0) {
      return { valid: false, error: `${key} must be a required finite nonnegative integer` };
    }
  }
  const generated = s.total_windows_generated as number;
  const evaluated = s.total_windows_evaluated as number;
  const dropped = s.total_windows_dropped as number;

  // Arithmetic on evaluated + dropped:
  if (s.status === 'error') {
    // For error stop ACKs, permit: evaluated + dropped <= generated
    if (evaluated + dropped > generated) {
      return {
        valid: false,
        error: `Error stop ACK evaluated (${evaluated}) + dropped (${dropped}) exceeds generated (${generated})`,
      };
    }
  } else {
    // For non-error stop ACKs, evaluated + dropped must equal generated
    if (evaluated + dropped !== generated) {
      return {
        valid: false,
        error: `total_windows_evaluated (${evaluated}) + total_windows_dropped (${dropped}) !== total_windows_generated (${generated})`,
      };
    }
  }

  // Reliability counters:
  for (const key of [
    'reliable_window_count',
    'limited_reliability_window_count',
    'insufficient_reliability_window_count',
    'degraded_quality_window_count',
  ] as const) {
    const val = s[key];
    if (typeof val !== 'number' || !Number.isInteger(val) || val < 0) {
      return { valid: false, error: `${key} must be a required finite nonnegative integer` };
    }
  }
  const reliable = s.reliable_window_count as number;
  const limited = s.limited_reliability_window_count as number;
  const insufficient = s.insufficient_reliability_window_count as number;
  const degraded = s.degraded_quality_window_count as number;

  // Reliability counters must still sum exactly to generated for every status
  if (reliable + limited + insufficient !== generated) {
    return {
      valid: false,
      error: `Counters do not sum to total_windows_generated: ${reliable} + ${limited} + ${insufficient} !== ${generated}`,
    };
  }

  if (degraded < limited) {
    return {
      valid: false,
      error: `degraded_quality_window_count (${degraded}) cannot be less than limited_reliability_window_count (${limited})`,
    };
  }

  if (degraded > generated) {
    return { valid: false, error: 'degraded_quality_window_count exceeds total_windows_generated' };
  }

  if (typeof s.quality_origin_verdict_allowed !== 'boolean') {
    return { valid: false, error: 'quality_origin_verdict_allowed must be a required boolean' };
  }

  if (s.final_reason !== null && typeof s.final_reason !== 'string') {
    return { valid: false, error: 'final_reason must be string or null' };
  }

  // Status coherence:
  if (s.status === 'error') {
    if (s.quality_origin_verdict_allowed !== false) {
      return { valid: false, error: 'Error stop cannot permit origin verdict' };
    }
    // Accept any of the 4 real temporal DecisionStates.
  }

  if (degraded > 0 && s.quality_origin_verdict_allowed === true) {
    return {
      valid: false,
      error: 'quality_origin_verdict_allowed cannot be true when degraded_quality_window_count > 0',
    };
  }

  if (s.status === 'success') {
    if (s.quality_origin_verdict_allowed !== true) {
      return { valid: false, error: 'status success requires quality_origin_verdict_allowed=true' };
    }
    if (limited > 0 || degraded > 0) {
      return { valid: false, error: 'status success cannot have limited or degraded windows' };
    }
    if (reliable <= 0 || evaluated <= 0) {
      return { valid: false, error: 'status success requires at least one reliable and evaluated window' };
    }
    const dec = (s.final_decision_state as string).toLowerCase();
    if (dec !== 'low' && dec !== 'uncertain' && dec !== 'high') {
      return { valid: false, error: `status success requires low, uncertain, or high final_decision_state, got ${dec}` };
    }
  }

  if (s.status === 'limited_reliability') {
    if (s.quality_origin_verdict_allowed !== false) {
      return { valid: false, error: 'status limited_reliability requires quality_origin_verdict_allowed=false' };
    }
    if ((s.final_decision_state as string).toLowerCase() !== 'uncertain') {
      return { valid: false, error: 'status limited_reliability requires final_decision_state=uncertain' };
    }
  }

  if (s.status === 'insufficient_evidence') {
    if (s.quality_origin_verdict_allowed !== false) {
      return { valid: false, error: 'status insufficient_evidence requires quality_origin_verdict_allowed=false' };
    }
    if ((s.final_decision_state as string).toLowerCase() !== 'insufficient_evidence') {
      return { valid: false, error: 'status insufficient_evidence requires final_decision_state=insufficient_evidence' };
    }
  }

  return { valid: true, stopAck: raw as unknown as ServerStopAck };
}

export function validateAggregatePermissionAndStatusCoherence(args: {
  status: string;
  quality_origin_verdict_allowed: boolean;
  reliable_window_count: number;
  total_windows_evaluated?: number;
  limited_reliability_window_count: number;
  degraded_quality_window_count: number;
  final_decision_state: string;
}): { valid: true; originVerdictPermitted: boolean } | { valid: false; error: string } {
  const {
    status,
    quality_origin_verdict_allowed,
    reliable_window_count,
    total_windows_evaluated,
    limited_reliability_window_count,
    degraded_quality_window_count,
    final_decision_state,
  } = args;

  const recognizedDecisionStates: readonly DecisionState[] = [
    'low',
    'uncertain',
    'high',
    'insufficient_evidence',
  ];
  if (!recognizedDecisionStates.includes(final_decision_state.toLowerCase() as DecisionState)) {
    return { valid: false, error: `Unrecognized final_decision_state: ${final_decision_state}` };
  }

  // Reject contradictory combinations:
  if (status === 'limited_reliability' && quality_origin_verdict_allowed) {
    return { valid: false, error: 'Contradiction: limited_reliability cannot permit origin verdict' };
  }
  if (status === 'insufficient_evidence' && quality_origin_verdict_allowed) {
    return { valid: false, error: 'Contradiction: insufficient_evidence cannot permit origin verdict' };
  }
  if (status === 'error' && quality_origin_verdict_allowed) {
    return { valid: false, error: 'Contradiction: error status cannot permit origin verdict' };
  }
  if (status === 'unsupported' && quality_origin_verdict_allowed) {
    return { valid: false, error: 'Contradiction: unsupported status cannot permit origin verdict' };
  }

  if (status === 'success' && quality_origin_verdict_allowed) {
    if (limited_reliability_window_count > 0) {
      return { valid: false, error: 'Contradiction: success with permission=true cannot have limited windows' };
    }
    if (degraded_quality_window_count > 0) {
      return { valid: false, error: 'Contradiction: success with permission=true cannot have degraded quality windows' };
    }
    const evaluatedCount = total_windows_evaluated !== undefined ? total_windows_evaluated : reliable_window_count;
    if (reliable_window_count <= 0 || evaluatedCount <= 0) {
      return { valid: false, error: 'Contradiction: success with permission=true requires at least one reliable/evaluated window' };
    }
  }

  const originVerdictPermitted =
    status === 'success' &&
    quality_origin_verdict_allowed === true &&
    reliable_window_count > 0 &&
    (total_windows_evaluated === undefined || total_windows_evaluated > 0) &&
    limited_reliability_window_count === 0 &&
    degraded_quality_window_count === 0 &&
    (final_decision_state.toLowerCase() === 'low' ||
      final_decision_state.toLowerCase() === 'uncertain' ||
      final_decision_state.toLowerCase() === 'high');

  return { valid: true, originVerdictPermitted };
}

export function validateServerWindowResult(raw: unknown):
  | { valid: true; windowResult: ServerWindowResult }
  | { valid: false; error: string } {
  if (!raw || typeof raw !== 'object') {
    return { valid: false, error: 'window_result must be a non-null object' };
  }
  const e = raw as Record<string, unknown>;

  const requiredKeys = [
    'type',
    'session_id',
    'window_index',
    'window_start_sec',
    'window_end_sec',
    'speech_seconds',
    'is_eligible',
    'synthetic_score',
    'uncalibrated_synthetic_score',
    'score_is_calibrated',
    'operating_threshold',
    'threshold_low',
    'threshold_high',
    'operating_threshold_source',
    'raw_score',
    'calibrated_score',
    'decision',
    'input_reliability',
    'quality_status',
    'abstain_reason',
  ] as const;

  for (const k of requiredKeys) {
    if (!hasKey(e, k)) {
      return { valid: false, error: `window_result missing required key: ${k}` };
    }
  }

  if (e.type !== 'window_result') {
    return { valid: false, error: `type must be window_result, got ${String(e.type)}` };
  }

  if (typeof e.session_id !== 'string' || e.session_id.trim().length === 0) {
    return { valid: false, error: 'session_id must be a non-empty string' };
  }

  if (typeof e.window_index !== 'number' || !Number.isInteger(e.window_index) || e.window_index < 0) {
    return { valid: false, error: 'window_index must be a non-negative integer' };
  }

  if (typeof e.window_start_sec !== 'number' || !Number.isFinite(e.window_start_sec) || e.window_start_sec < 0) {
    return { valid: false, error: 'window_start_sec must be a non-negative finite number' };
  }

  if (typeof e.window_end_sec !== 'number' || !Number.isFinite(e.window_end_sec) || e.window_end_sec < e.window_start_sec) {
    return { valid: false, error: 'window_end_sec must be a finite number >= window_start_sec' };
  }

  if (
    typeof e.speech_seconds !== 'number' ||
    !Number.isFinite(e.speech_seconds) ||
    e.speech_seconds < 0 ||
    e.speech_seconds > ((e.window_end_sec as number) - (e.window_start_sec as number) + 1e-4)
  ) {
    return { valid: false, error: 'speech_seconds must be a finite non-negative number within window duration' };
  }

  if (typeof e.is_eligible !== 'boolean') {
    return { valid: false, error: 'is_eligible must be a boolean' };
  }

  if (typeof e.score_is_calibrated !== 'boolean') {
    return { valid: false, error: 'score_is_calibrated must be a boolean' };
  }

  for (const tKey of ['operating_threshold', 'threshold_low', 'threshold_high'] as const) {
    const tVal = e[tKey];
    if (typeof tVal !== 'number' || !Number.isFinite(tVal)) {
      return { valid: false, error: `${tKey} must be a finite number` };
    }
  }

  if (typeof e.operating_threshold_source !== 'string' || e.operating_threshold_source.trim().length === 0) {
    return { valid: false, error: 'operating_threshold_source must be a non-empty string' };
  }

  if (e.abstain_reason !== null && typeof e.abstain_reason !== 'string') {
    return { valid: false, error: 'abstain_reason must be null or string' };
  }

  // exact quality and reliability validators pass
  const qVal = validateQualityMetrics(e.quality_status);
  if (!qVal.valid) {
    return { valid: false, error: `Invalid quality_status: ${qVal.error}` };
  }

  const relVal = validateInputReliability(e.input_reliability);
  if (!relVal.valid) {
    return { valid: false, error: `Invalid input_reliability: ${relVal.error}` };
  }
  const rel = relVal.value;
  const isReliable = rel.state === 'reliable';

  if (e.is_eligible !== isReliable) {
    return {
      valid: false,
      error: `is_eligible (${String(e.is_eligible)}) is not coherent with input_reliability state (${rel.state})`,
    };
  }

  // fully validate TemporalDecision matching enclosing window
  const decVal = validateTemporalDecision(e.decision, {
    window_index: e.window_index as number,
    window_start_sec: e.window_start_sec as number,
    window_end_sec: e.window_end_sec as number,
    speech_seconds: e.speech_seconds as number,
  });
  if (!decVal.valid) {
    return { valid: false, error: `Invalid decision: ${decVal.error}` };
  }
  const dec = decVal.value;

  // The backend serializes these values both at window level and inside TemporalDecision.
  // Reject contradictory duplicates before either representation can enter React state.
  if (!Object.is(e.synthetic_score, dec.synthetic_score)) {
    return { valid: false, error: 'synthetic_score does not match decision.synthetic_score' };
  }
  if (e.score_is_calibrated !== dec.score_is_calibrated) {
    return { valid: false, error: 'score_is_calibrated does not match decision.score_is_calibrated' };
  }
  if (!Object.is(e.calibrated_score, dec.calibrated_score)) {
    return { valid: false, error: 'calibrated_score does not match decision.calibrated_score' };
  }

  // Coherence:
  if (isReliable) {
    if (dec.state !== 'low' && dec.state !== 'uncertain' && dec.state !== 'high') {
      return { valid: false, error: `Reliable window decision.state must be low, uncertain, or high; got ${dec.state}` };
    }
    if (typeof e.synthetic_score !== 'number' || !Number.isFinite(e.synthetic_score) || e.synthetic_score < 0 || e.synthetic_score > 1.0) {
      return { valid: false, error: 'Reliable window requires finite synthetic_score in [0, 1]' };
    }
    if (typeof e.uncalibrated_synthetic_score !== 'number' || !Number.isFinite(e.uncalibrated_synthetic_score) || e.uncalibrated_synthetic_score < 0 || e.uncalibrated_synthetic_score > 1.0) {
      return { valid: false, error: 'Reliable window requires finite uncalibrated_synthetic_score in [0, 1]' };
    }
    if (typeof e.raw_score !== 'number' || !Number.isFinite(e.raw_score) || e.raw_score < 0 || e.raw_score > 1.0) {
      return { valid: false, error: 'Reliable window requires finite raw_score in [0, 1]' };
    }
    if (
      !Object.is(e.synthetic_score, e.uncalibrated_synthetic_score) ||
      !Object.is(e.synthetic_score, e.raw_score)
    ) {
      return { valid: false, error: 'Reliable window score aliases must match synthetic_score' };
    }
    if (e.calibrated_score !== null && (typeof e.calibrated_score !== 'number' || !Number.isFinite(e.calibrated_score))) {
      return { valid: false, error: 'calibrated_score must be null or finite number' };
    }
    if (dec.synthetic_score === null) {
      return { valid: false, error: 'Reliable window decision.synthetic_score must be non-null' };
    }
  } else {
    // non-reliable ineligible window
    if (dec.state !== 'uncertain' && dec.state !== 'insufficient_evidence') {
      return { valid: false, error: `Non-reliable window decision.state must be uncertain or insufficient_evidence; got ${dec.state}` };
    }
    if (e.synthetic_score !== null) {
      return { valid: false, error: 'Non-reliable window must have null synthetic_score' };
    }
    if (e.uncalibrated_synthetic_score !== null) {
      return { valid: false, error: 'Non-reliable window must have null uncalibrated_synthetic_score' };
    }
    if (e.raw_score !== null) {
      return { valid: false, error: 'Non-reliable window must have null raw_score' };
    }
    if (e.calibrated_score !== null) {
      return { valid: false, error: 'Non-reliable window must have null calibrated_score' };
    }
    if (dec.synthetic_score !== null) {
      return { valid: false, error: 'Non-reliable window decision.synthetic_score must be null' };
    }
    if (dec.calibrated_score !== null) {
      return { valid: false, error: 'Non-reliable window decision.calibrated_score must be null' };
    }
  }

  return { valid: true, windowResult: raw as unknown as ServerWindowResult };
}

export function getWebSocketStreamUrl(): string {
  if (typeof window === 'undefined') {
    return 'ws://127.0.0.1:5173/api/v1/stream';
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/api/v1/stream`;
}

export function fileWindowState(win: FileInferenceResult['windows'][number]): {
  label: string;
  className: string;
} {
  const decState = (win.decision_state || win.decision?.state) as string | undefined;
  if (
    win.is_eligible === false ||
    win.input_reliability?.origin_verdict_allowed === false ||
    win.input_reliability?.state === 'limited' ||
    win.input_reliability?.state === 'insufficient' ||
    decState === 'insufficient_evidence' ||
    decState === 'ineligible' ||
    decState === 'limited_reliability'
  ) {
    if (win.input_reliability?.state === 'limited' || decState === 'limited_reliability') {
      return { label: 'LIMITED', className: 'badge-limited' };
    }
    return { label: 'ABSTAINED', className: 'badge-insufficient_evidence' };
  }
  switch (decState) {
    case 'high':
      return { label: 'ELEVATED', className: 'badge-high' };
    case 'uncertain':
      return { label: 'UNCERTAIN', className: 'badge-uncertain' };
    case 'low':
      return { label: 'LOW ANOMALY', className: 'badge-low' };
    default:
      return { label: 'UNKNOWN', className: 'badge-unknown' };
  }
}

export function resolveFileDecisionState(result: FileInferenceResult): DetectionStatus {
  if (result.status === 'limited_reliability') {
    return 'limited_reliability';
  }
  if (result.status === 'insufficient_evidence') {
    return 'insufficient_evidence';
  }
  if (result.status === 'error' || result.status === 'unsupported') {
    return 'error';
  }
  if (
    result.status === 'success' &&
    result.quality_origin_verdict_allowed === true &&
    result.reliable_window_count > 0 &&
    result.limited_reliability_window_count === 0 &&
    result.degraded_quality_window_count === 0
  ) {
    const dec = result.final_decision_state ? result.final_decision_state.toLowerCase() : '';
    if (dec === 'high' || dec === 'low' || dec === 'uncertain') {
      return dec as DetectionStatus;
    }
  }
  if (result.limited_reliability_window_count > 0 || result.degraded_quality_window_count > 0) {
    return 'limited_reliability';
  }
  return 'insufficient_evidence';
}

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'stream' | 'file'>('stream');
  const [readiness, setReadiness] = useState<SystemReadiness>({
    status: 'checking',
    is_mock_detector: false,
  });
  const [isMockDetector, setIsMockDetector] = useState<boolean>(false);

  // Streaming State
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [recordingState, setRecordingState] = useState<RecordingState>('idle');
  const [status, setStatus] = useState<DetectionStatus>('insufficient_evidence');
  const [reasonDetails, setReasonDetails] = useState<string>(
    'Stream active speech to begin acoustic origin assessment.'
  );
  const [uncalibratedScore, setUncalibratedScore] = useState<number | null>(null);
  const [smoothedScore, setSmoothedScore] = useState<number | null>(null);
  const [thresholdLow, setThresholdLow] = useState<number>(0.35);
  const [thresholdHigh, setThresholdHigh] = useState<number>(0.65);
  const [operatingThresholdSource, setOperatingThresholdSource] = useState<string>('unvalidated_heuristic');
  const [activeSpeechSeconds, setActiveSpeechSeconds] = useState<number>(0);
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetrics>({
    isClipping: false,
    isSilent: false,
    hasTimingGap: false,
    clippingCount: 0,
    gapCount: 0,
  });
  const [timeline, setTimeline] = useState<WindowHistoryItem[]>([]);
  const [stopSummary, setStopSummary] = useState<ServerStopAck | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [liveOriginVerdictAllowed, setLiveOriginVerdictAllowed] = useState<boolean>(false);
  const [sessionCounters, setSessionCounters] = useState({
    total: 0,
    reliable: 0,
    limited: 0,
    insufficient: 0,
    degraded: 0,
  });

  // File Analysis State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileHopSeconds, setFileHopSeconds] = useState<number>(1.0);
  const [fileThreshold, setFileThreshold] = useState<number>(0.65);
  const [isFileAnalyzing, setIsFileAnalyzing] = useState<boolean>(false);
  const [fileResult, setFileResult] = useState<FileInferenceResult | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);

  // Web Audio & WebSocket References
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const silentGainRef = useRef<GainNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const mountedRef = useRef<boolean>(true);
  const stopSentRef = useRef<boolean>(false);
  const stopTimerRef = useRef<number | null>(null);

  // Check system readiness
  useEffect(() => {
    let isMounted = true;
    async function checkReadiness(): Promise<void> {
      try {
        const resp = await fetch('/api/v1/ready');
        if (resp.ok) {
          const data = (await resp.json()) as { is_mock_detector: boolean; device?: string; config_hash?: string; status?: string };
          if (isMounted) {
            setReadiness({
              status: 'ready',
              is_mock_detector: Boolean(data.is_mock_detector),
              device: data.device,
              config_hash: data.config_hash ?? null,
            });
            setIsMockDetector(Boolean(data.is_mock_detector));
          }
        } else if (resp.status === 503) {
          const data = (await resp.json()) as { detail?: { reason?: string; message?: string } };
          if (isMounted) {
            setReadiness({
              status: 'not_ready',
              is_mock_detector: false,
              reason:
                data.detail?.reason ??
                data.detail?.message ??
                'Checkpoint absent or detector not initialized.',
            });
          }
        } else {
          if (isMounted) {
            setReadiness({
              status: 'error',
              is_mock_detector: false,
              reason: `VoiceShield readiness request failed with HTTP ${resp.status}.`,
            });
          }
        }
      } catch {
        if (isMounted) {
          setReadiness({
            status: 'error',
            is_mock_detector: false,
            reason: 'VoiceShield API dev server not reached at /api. Start with "make api".',
          });
        }
      }
    }
    void checkReadiness();
    return () => {
      isMounted = false;
    };
  }, []);

  const stopAudioCapture = useCallback((): void => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (workletNodeRef.current) {
      try {
        workletNodeRef.current.disconnect();
      } catch {
        // Audio graph teardown is best effort.
      }
      workletNodeRef.current = null;
    }
    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.disconnect();
      } catch {
        // Audio graph teardown is best effort.
      }
      sourceNodeRef.current = null;
    }
    if (silentGainRef.current) {
      try {
        silentGainRef.current.disconnect();
      } catch {
        // Audio graph teardown is best effort.
      }
      silentGainRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      void audioContextRef.current.close();
      audioContextRef.current = null;
    }
  }, []);

  const forceCleanup = useCallback((): void => {
    if (stopTimerRef.current !== null) {
      window.clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    try {
      stopAudioCapture();
      const ws = wsRef.current;
      wsRef.current = null;
      stopSentRef.current = false;
      if (ws && ws.readyState !== WebSocket.CLOSED) {
        try {
          ws.close(1000, 'Client cleanup');
        } catch {
          // Cleanup is best effort.
        }
      }
    } finally {
      if (mountedRef.current) {
        setRecordingState('idle');
        setConnectionState('disconnected');
      }
    }
  }, [stopAudioCapture]);

  const stopAudioAndDisconnect = useCallback((): void => {
    if (stopSentRef.current) return;
    stopAudioCapture();
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      forceCleanup();
      return;
    }
    stopSentRef.current = true;
    setRecordingState('stopping');
    try {
      ws.send(JSON.stringify({ type: 'stop' }));
    } catch {
      forceCleanup();
      return;
    }
    stopTimerRef.current = window.setTimeout(() => {
      forceCleanup();
      setRecordingState('idle');
      setConnectionState('disconnected');
    }, 1800);
  }, [forceCleanup, stopAudioCapture]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      forceCleanup();
    };
  }, [forceCleanup]);

  const startStream = async (): Promise<void> => {
    setErrorMessage(null);
    setStopSummary(null);
    setSessionCounters({ total: 0, reliable: 0, limited: 0, insufficient: 0, degraded: 0 });
    setTimeline([]);
    setActiveSpeechSeconds(0);
    setStatus('insufficient_evidence');
    setReasonDetails('Stream active speech to begin acoustic origin assessment.');
    setUncalibratedScore(null);
    setSmoothedScore(null);
    setLiveOriginVerdictAllowed(false);
    setQualityMetrics({
      isClipping: false,
      isSilent: false,
      hasTimingGap: false,
      clippingCount: 0,
      gapCount: 0,
    });
    setRecordingState('requesting_permission');
    setConnectionState('connecting');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      mediaStreamRef.current = stream;

      const AudioCtxClass =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioCtxClass();
      audioContextRef.current = audioCtx;
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      try {
        await audioCtx.audioWorklet.addModule('/pcm-worklet-processor.js');
      } catch {
        const workletCode = `
          class PCMWorkletProcessor extends AudioWorkletProcessor {
            constructor() {
              super();
              this.bufferSize = 1024;
              this.buffer = new Int16Array(this.bufferSize);
              this.bufferIndex = 0;
            }

            process(inputs, outputs, parameters) {
              const input = inputs[0];
              if (!input || !input[0]) return true;
              const channelData = input[0];

              for (let i = 0; i < channelData.length; i++) {
                let s = channelData[i];
                if (s > 1.0) s = 1.0;
                else if (s < -1.0) s = -1.0;
                this.buffer[this.bufferIndex++] = s < 0 ? s * 32768 : s * 32767;

                if (this.bufferIndex >= this.bufferSize) {
                  this.port.postMessage(this.buffer.buffer.slice(0));
                  this.bufferIndex = 0;
                }
              }
              return true;
            }
          }
          registerProcessor('pcm-worklet-processor', PCMWorkletProcessor);
        `;
        const blob = new Blob([workletCode], { type: 'application/javascript' });
        const blobUrl = URL.createObjectURL(blob);
        try {
          await audioCtx.audioWorklet.addModule(blobUrl);
        } finally {
          URL.revokeObjectURL(blobUrl);
        }
      }

      const sourceNode = audioCtx.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioCtx, 'pcm-worklet-processor');
      const silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      workletNodeRef.current = workletNode;
      sourceNodeRef.current = sourceNode;
      silentGainRef.current = silentGain;
      sourceNode.connect(workletNode);
      workletNode.connect(silentGain);
      silentGain.connect(audioCtx.destination);

      const wsUrl = getWebSocketStreamUrl();
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        setRecordingState('recording');
        const startPayload = {
          type: 'start',
          protocol_version: '1.0',
          sample_rate: audioCtx.sampleRate,
          channels: 1,
          pcm_encoding: 'int16',
          metadata: { client: 'voiceshield-web-demo' },
        };
        ws.send(JSON.stringify(startPayload));
      };

      workletNode.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(event.data);
        }
      };

      ws.onmessage = (event: MessageEvent) => {
        let raw: Record<string, unknown>;
        try {
          raw = JSON.parse(event.data as string) as Record<string, unknown>;
          if (!raw || typeof raw !== 'object') return;
        } catch {
          // Genuinely non-JSON transport noise: ignore safely
          return;
        }

        const msgType = raw.type;
        if (typeof msgType !== 'string') return;

        if (msgType === 'session_started') {
          try {
            const data = raw as unknown as ServerStartAck;
            setConnectionState('connected');
            if (data.is_mock_detector !== undefined) {
              setIsMockDetector(Boolean(data.is_mock_detector));
            }
          } catch (err: unknown) {
            setStatus('error');
            const msg = err instanceof Error ? err.message : String(err);
            setErrorMessage(`Contract Error: Malformed session_started: ${msg}`);
            setReasonDetails(`Contract Error: Malformed session_started: ${msg}`);
            setUncalibratedScore(null);
            setSmoothedScore(null);
            setLiveOriginVerdictAllowed(false);
          }
        } else if (msgType === 'quality_update') {
          try {
            const data = raw as unknown as ServerQualityEvent;
            setQualityMetrics({
              isClipping: Boolean(data.is_clipped),
              isSilent: Boolean(data.is_silent),
              hasTimingGap: Boolean(data.has_timing_gap),
              clippingCount: typeof data.clipping_count === 'number' ? data.clipping_count : 0,
              gapCount: typeof data.gap_count === 'number' ? data.gap_count : 0,
            });
          } catch {
            // Non-fatal telemetry update
          }
        } else if (msgType === 'window_result') {
          try {
            const winVal = validateServerWindowResult(raw);
            if (!winVal.valid) {
              setStatus('error');
              setErrorMessage(`Contract Error: Invalid window_result frame: ${winVal.error}`);
              setUncalibratedScore(null);
              setSmoothedScore(null);
              setLiveOriginVerdictAllowed(false);
              setReasonDetails(
                `Contract validation failed on window_result: ${winVal.error}. Origin verdict withheld.`
              );
              return;
            }

            const data = winVal.windowResult;
            const rel = data.input_reliability;
            setActiveSpeechSeconds(data.speech_seconds);

            // Maintain running session counters independently of the 20-item visual timeline
            setSessionCounters((prev) => ({
              total: prev.total + 1,
              reliable: prev.reliable + (rel.state === 'reliable' ? 1 : 0),
              limited: prev.limited + (rel.state === 'limited' ? 1 : 0),
              insufficient: prev.insufficient + (rel.state === 'insufficient' ? 1 : 0),
              degraded: prev.degraded + (rel.reasons.some((r) => LIMITED_REASONS.has(r)) ? 1 : 0),
            }));

            const isPermitted =
              data.is_eligible === true &&
              rel.state === 'reliable' &&
              rel.origin_verdict_allowed === true;

            setLiveOriginVerdictAllowed(isPermitted);

            if (isPermitted) {
              const decState = data.decision.state.toLowerCase() as DetectionStatus;
              setStatus(decState);
              setReasonDetails(data.decision.reason_details);

              const score =
                data.synthetic_score ??
                data.uncalibrated_synthetic_score ??
                data.raw_score ??
                null;
              setUncalibratedScore(score);
              const smoothed = data.decision.smoothed_score ?? null;
              setSmoothedScore(smoothed);
            } else {
              // Gated window: withhold origin verdict and clear scores
              if (rel.state === 'limited') {
                setStatus('limited_reliability');
                setReasonDetails(
                  rel.explanation ||
                    `Inference withheld: input audio quality is degraded (${rel.reasons.join(', ')}).`
                );
              } else {
                setStatus('insufficient_evidence');
                setReasonDetails(
                  rel.explanation ||
                    `Inference withheld: insufficient speech or silence (${rel.reasons.join(', ')}).`
                );
              }
              setUncalibratedScore(null);
              setSmoothedScore(null);
            }

            if (data.threshold_low !== undefined) {
              setThresholdLow(data.threshold_low);
            }
            if (data.threshold_high !== undefined) {
              setThresholdHigh(data.threshold_high);
            } else if (data.operating_threshold !== undefined) {
              setThresholdHigh(data.operating_threshold);
            }
            if (data.operating_threshold_source !== undefined) {
              setOperatingThresholdSource(data.operating_threshold_source);
            }

            const historyItem: WindowHistoryItem = {
              windowIndex: data.window_index,
              windowStartSec: data.window_start_sec,
              windowEndSec: data.window_end_sec,
              speechSeconds: data.speech_seconds,
              isEligible: isPermitted,
              syntheticScore: isPermitted
                ? (data.synthetic_score ?? data.uncalibrated_synthetic_score ?? data.raw_score ?? null)
                : null,
              scoreIsCalibrated: Boolean(data.score_is_calibrated),
              calibratedScore: isPermitted ? (data.calibrated_score ?? null) : null,
              rawScore: isPermitted ? (data.raw_score ?? data.synthetic_score ?? null) : null,
              smoothedScore: isPermitted ? (data.decision.smoothed_score ?? null) : null,
              thresholdLow: data.threshold_low ?? thresholdLow,
              thresholdHigh: data.threshold_high ?? data.operating_threshold ?? thresholdHigh,
              status: isPermitted
                ? (data.decision.state.toLowerCase() as DetectionStatus)
                : (rel.state === 'limited' ? 'limited_reliability' : 'insufficient_evidence'),
              reason: data.decision.reason,
              reasonDetails: isPermitted ? data.decision.reason_details : (rel.explanation || 'Inference withheld'),
              timestamp: new Date().toLocaleTimeString(),
              inputReliability: rel,
              qualityStatus: data.quality_status,
              abstainReason: data.abstain_reason ?? (!rel.origin_verdict_allowed ? rel.reasons.join(', ') : null),
              inferencePerformed: isPermitted,
            };

            setTimeline((prev) => [historyItem, ...prev.slice(0, 19)]);
          } catch (err: unknown) {
            setStatus('error');
            const msg = err instanceof Error ? err.message : String(err);
            setErrorMessage(`Contract Error: Malformed window_result: ${msg}`);
            setReasonDetails(`Contract validation failed on window_result: ${msg}. Origin verdict withheld.`);
            setUncalibratedScore(null);
            setSmoothedScore(null);
            setLiveOriginVerdictAllowed(false);
          }
        } else if (msgType === 'session_stopped') {
          try {
            const stopVal = validateStreamingStopAck(raw);
            if (!stopVal.valid) {
              setStatus('error');
              setErrorMessage(`Contract Error: Invalid session_stopped frame: ${stopVal.error}`);
              setReasonDetails(
                `Contract validation failed on session_stopped: ${stopVal.error}. Origin verdict withheld.`
              );
              setUncalibratedScore(null);
              setSmoothedScore(null);
              setLiveOriginVerdictAllowed(false);
              forceCleanup();
              return;
            }

            const stopAck = stopVal.stopAck;

            // Replace live counters with authoritative final ACK counters
            setSessionCounters({
              total: stopAck.total_windows_generated,
              reliable: stopAck.reliable_window_count,
              limited: stopAck.limited_reliability_window_count,
              insufficient: stopAck.insufficient_reliability_window_count,
              degraded: stopAck.degraded_quality_window_count,
            });
            setStopSummary(stopAck);

            if (stopAck.status === 'error') {
              setStatus('error');
              setErrorMessage(stopAck.final_reason ?? 'Session stopped with error');
              setReasonDetails(stopAck.final_reason ?? 'Session stopped with error. Origin verdict withheld.');
              setUncalibratedScore(null);
              setSmoothedScore(null);
              setLiveOriginVerdictAllowed(false);
            } else if (stopAck.status === 'success' && stopAck.quality_origin_verdict_allowed === true) {
              const finalState = stopAck.final_decision_state.toLowerCase() as DetectionStatus;
              setStatus(finalState);
              setLiveOriginVerdictAllowed(true);
              if (stopAck.final_reason) {
                setReasonDetails(stopAck.final_reason);
              }
            } else if (stopAck.status === 'limited_reliability') {
              setStatus('limited_reliability');
              setLiveOriginVerdictAllowed(false);
              setReasonDetails(stopAck.final_reason ?? 'LIMITED RELIABILITY — ORIGIN VERDICT WITHHELD');
              setUncalibratedScore(null);
              setSmoothedScore(null);
            } else {
              setStatus('insufficient_evidence');
              setLiveOriginVerdictAllowed(false);
              setReasonDetails(stopAck.final_reason ?? 'INSUFFICIENT EVIDENCE — ORIGIN VERDICT WITHHELD');
              setUncalibratedScore(null);
              setSmoothedScore(null);
            }
            forceCleanup();
          } catch (err: unknown) {
            setStatus('error');
            const msg = err instanceof Error ? err.message : String(err);
            setErrorMessage(`Contract Error: Malformed session_stopped: ${msg}`);
            setReasonDetails(`Contract validation failed on session_stopped: ${msg}. Origin verdict withheld.`);
            setUncalibratedScore(null);
            setSmoothedScore(null);
            setLiveOriginVerdictAllowed(false);
            forceCleanup();
          }
        } else if (msgType === 'error') {
          try {
            const data = raw as unknown as { error_code?: string; message?: string };
            const code = typeof data.error_code === 'string' ? data.error_code : 'SERVER_ERROR';
            const msg = typeof data.message === 'string' ? data.message : 'Unknown server error';
            setStatus('error');
            setErrorMessage(`${code}: ${msg}`);
            setReasonDetails(msg);
            setUncalibratedScore(null);
            setSmoothedScore(null);
            setLiveOriginVerdictAllowed(false);
          } catch {
            setStatus('error');
            setErrorMessage('SERVER_ERROR: Malformed error frame');
            setReasonDetails('Malformed error frame received from server.');
            setUncalibratedScore(null);
            setSmoothedScore(null);
            setLiveOriginVerdictAllowed(false);
          }
        }
      };

      ws.onerror = () => {
        if (!mountedRef.current) return;
        setConnectionState('error');
        setStatus('error');
        setErrorMessage('WebSocket connection error. Is the VoiceShield backend running?');
        forceCleanup();
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setConnectionState('disconnected');
        setRecordingState('idle');
        wsRef.current = null;
      };
    } catch (err: unknown) {
      const errorMsg =
        err instanceof Error ? err.message : 'Microphone permission denied or audio device error.';
      setErrorMessage(errorMsg);
      setRecordingState('error');
      setConnectionState('error');
      setStatus('error');
      forceCleanup();
    }
  };

  const handleFileAnalysis = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsFileAnalyzing(true);
    setFileError(null);
    setFileResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const url = `/api/v1/analyze/file?hop_seconds=${fileHopSeconds}&threshold=${fileThreshold}`;
      const resp = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const errJson = (await resp.json()) as { detail?: { message?: string; error_code?: string } };
        throw new Error(
          errJson.detail?.message || `File analysis failed with HTTP status ${resp.status}`
        );
      }

      const rawResult = (await resp.json()) as Record<string, unknown>;

      // 1. Context-aware validation of file aggregate and all windows
      const fileVal = validateFileAggregateAndWindows(rawResult);
      if (!fileVal.valid) {
        setFileError(`Contract Error: ${fileVal.error}`);
        setFileResult(null);
        return;
      }
      const fileData = fileVal.file;

      // 2. Validate exact status
      const validStatuses: FileInferenceStatus[] = [
        'success',
        'limited_reliability',
        'insufficient_evidence',
        'error',
        'unsupported',
      ];
      if (!validStatuses.includes(fileData.status)) {
        setFileError(`Contract Error: Invalid file status: ${String(fileData.status)}`);
        setFileResult(null);
        return;
      }

      // 3. Validate permission boolean
      if (typeof fileData.quality_origin_verdict_allowed !== 'boolean') {
        setFileError('Contract Error: quality_origin_verdict_allowed must be a boolean');
        setFileResult(null);
        return;
      }

      // 4. Validate final_decision_state
      const recognizedDecisionStates: readonly DecisionState[] = [
        'low',
        'uncertain',
        'high',
        'insufficient_evidence',
      ];
      if (
        !fileData.final_decision_state ||
        !recognizedDecisionStates.includes(fileData.final_decision_state.toLowerCase() as DecisionState)
      ) {
        setFileError(`Contract Error: Invalid final_decision_state: ${String(fileData.final_decision_state)}`);
        setFileResult(null);
        return;
      }

      // 5. Validate aggregate coherence
      const coh = validateAggregatePermissionAndStatusCoherence({
        status: fileData.status,
        quality_origin_verdict_allowed: fileData.quality_origin_verdict_allowed,
        reliable_window_count: fileData.reliable_window_count,
        limited_reliability_window_count: fileData.limited_reliability_window_count,
        degraded_quality_window_count: fileData.degraded_quality_window_count,
        final_decision_state: fileData.final_decision_state.toLowerCase(),
      });
      if (!coh.valid) {
        setFileError(`Contract Error: Incoherent file result: ${coh.error}`);
        setFileResult(null);
        return;
      }

      setFileResult(fileData);
      if (fileData.is_mock_detector !== undefined) {
        setIsMockDetector(Boolean(fileData.is_mock_detector));
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Audio file analysis failed.';
      setFileError(msg);
      setFileResult(null);
    } finally {
      setIsFileAnalyzing(false);
    }
  };

  const isRecording = recordingState === 'recording' || recordingState === 'stopping';

  const streamReliableCount = sessionCounters.reliable;
  const streamLimitedCount = sessionCounters.limited;
  const streamInsufficientCount = sessionCounters.insufficient;
  const streamDegradedCount = sessionCounters.degraded;
  const streamOriginAllowed = stopSummary
    ? Boolean(stopSummary.quality_origin_verdict_allowed)
    : (sessionCounters.total > 0 && sessionCounters.reliable === sessionCounters.total && liveOriginVerdictAllowed);

  const fileVerdictPermitted = fileResult
    ? (fileResult.status === 'success' &&
       fileResult.quality_origin_verdict_allowed === true &&
       fileResult.reliable_window_count > 0 &&
       fileResult.limited_reliability_window_count === 0 &&
       fileResult.degraded_quality_window_count === 0 &&
       (fileResult.final_decision_state?.toLowerCase() === 'low' ||
        fileResult.final_decision_state?.toLowerCase() === 'uncertain' ||
        fileResult.final_decision_state?.toLowerCase() === 'high'))
    : false;

  const fileMeanScoreLabel = fileResult && !fileVerdictPermitted && fileResult.reliable_window_count > 0
    ? 'Mean Score (Advisory Model Evidence From Reliable Windows)'
    : 'Mean Synthetic Score';

  const filePeakScoreLabel = fileResult && !fileVerdictPermitted && fileResult.reliable_window_count > 0
    ? 'Peak Score (Advisory Model Evidence From Reliable Windows)'
    : 'Peak Window Score';

  const streamScoreLabel = !streamOriginAllowed && streamReliableCount > 0
    ? 'Raw Model Score (Advisory Evidence from Reliable Windows)'
    : 'Raw uncalibrated window score';

  const runtimeInfo = (() => {
    if (readiness.status === 'checking') {
      return { label: 'Checking', stateClass: 'checking' };
    }
    if (readiness.status === 'not_ready' || readiness.status === 'error') {
      return { label: 'Unavailable', stateClass: 'unavailable' };
    }
    if (isMockDetector) {
      return { label: 'Mock Detector (Demo)', stateClass: 'mock' };
    }
    return { label: 'Non-Mock Detector', stateClass: 'non-mock' };
  })();

  const handleTabKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>): void => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      e.preventDefault();
      const nextTab = activeTab === 'stream' ? 'file' : 'stream';
      setActiveTab(nextTab);
      const nextId = nextTab === 'stream' ? 'tab-stream-btn' : 'tab-file-btn';
      document.getElementById(nextId)?.focus();
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault();
      const prevTab = activeTab === 'file' ? 'stream' : 'file';
      setActiveTab(prevTab);
      const prevId = prevTab === 'stream' ? 'tab-stream-btn' : 'tab-file-btn';
      document.getElementById(prevId)?.focus();
    } else if (e.key === 'Home') {
      e.preventDefault();
      setActiveTab('stream');
      document.getElementById('tab-stream-btn')?.focus();
    } else if (e.key === 'End') {
      e.preventDefault();
      setActiveTab('file');
      document.getElementById('tab-file-btn')?.focus();
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="brand">
          <div className="brand-icon" aria-hidden="true">
            🛡️
          </div>
          <div className="brand-text">
            <div className="brand-heading-row">
              <h1 className="brand-title">VoiceShield</h1>
              <span className="brand-badge">Advisory Acoustic Analysis</span>
            </div>
            <p className="brand-subtitle">
              Local Microphone Advisory Demo (Local input only — does not capture cellular telephony)
            </p>
          </div>
        </div>

        <div className="trust-strip" role="region" aria-label="System status">
          <div
            className={`status-pill pill-readiness-${readiness.status}`}
            id="readiness-state-pill"
            data-testid="readiness-state-pill"
            title={readiness.reason || undefined}
          >
            <span className="dot" aria-hidden="true" />
            <span>
              System:{' '}
              {readiness.status === 'ready'
                ? 'Ready'
                : readiness.status === 'checking'
                ? 'Checking...'
                : readiness.status === 'not_ready'
                ? 'Not Ready'
                : 'Unavailable'}
            </span>
          </div>

          <div
            className={`status-pill pill-runtime-${runtimeInfo.stateClass}`}
            id="detector-runtime-pill"
            data-testid="detector-runtime-pill"
          >
            <span className="dot" aria-hidden="true" />
            <span>Runtime: {runtimeInfo.label}</span>
          </div>

          <div className={`status-pill pill-${connectionState}`} id="connection-state-pill">
            <span className="dot" aria-hidden="true" />
            <span>Connection: {connectionState}</span>
          </div>

          <div className={`status-pill pill-${recordingState}`} id="recording-state-pill">
            <span className="dot" aria-hidden="true" />
            <span>Mic: {recordingState}</span>
          </div>
        </div>
      </header>

      {/* Model Not Ready Alert (if real mode without checkpoint) */}
      {(readiness.status === 'not_ready' || readiness.status === 'error') && (
        <div className="model-not-ready-banner" role="alert" id="model-not-ready-banner">
          <strong>
            {readiness.status === 'not_ready'
              ? '⚠️ DETECTOR MODEL NOT READY'
              : '⚠️ DETECTOR READINESS UNAVAILABLE'}
          </strong>
          <p>
            {readiness.status === 'not_ready'
              ? 'VoiceShield is running in real mode but the detector is not ready. Real mode refuses to mark model as ready without verified weights.'
              : 'VoiceShield could not verify detector readiness. Streaming and file analysis remain unavailable until the API responds successfully.'}
          </p>
          {readiness.reason && <pre>{readiness.reason}</pre>}
        </div>
      )}

      {/* Mock Detector Visible Indicator Badge */}
      {isMockDetector && (
        <div
          className="mock-detector-badge"
          data-testid="mock-detector-badge"
          role="status"
          id="mock-detector-badge"
        >
          <span className="mock-badge-tag">[MOCK DETECTOR - TEST MODE]</span>
          <span className="mock-badge-text">
            Synthetic weights active for UI and transport demonstration. Anomaly scores are mock values
            and do not evaluate deepfakes.
          </span>
        </div>
      )}

      {/* Navigation Tabs */}
      <nav className="demo-tabs-nav" aria-label="Analysis mode navigation">
        <div role="tablist" aria-label="Analysis Mode" className="demo-tabs">
          <button
            type="button"
            role="tab"
            className={`tab-btn ${activeTab === 'stream' ? 'active' : ''}`}
            onClick={() => setActiveTab('stream')}
            onKeyDown={handleTabKeyDown}
            id="tab-stream-btn"
            aria-selected={activeTab === 'stream'}
            aria-controls="panel-stream"
            tabIndex={activeTab === 'stream' ? 0 : -1}
          >
            <span aria-hidden="true">🎙️</span>
            <span>Live Microphone Stream</span>
          </button>
          <button
            type="button"
            role="tab"
            className={`tab-btn ${activeTab === 'file' ? 'active' : ''}`}
            onClick={() => setActiveTab('file')}
            onKeyDown={handleTabKeyDown}
            id="tab-file-btn"
            aria-selected={activeTab === 'file'}
            aria-controls="panel-file"
            tabIndex={activeTab === 'file' ? 0 : -1}
          >
            <span aria-hidden="true">📁</span>
            <span>File Audio Analysis</span>
          </button>
        </div>
      </nav>

      {/* Strict Compliance and Privacy Banner */}
      <div className="compliance-banner" id="compliance-banner">
        <div className="compliance-row">
          <span className="badge-tag">PRIVACY NOTICE:</span>
          <span>
            Audio is processed in volatile memory and immediately discarded. Audio is never stored in
            your browser or on the server.
          </span>
        </div>
        <div className="compliance-row">
          <span className="badge-tag">ADVISORY NOTICE:</span>
          <span>
            VoiceShield outputs acoustic anomaly indicators only. It does not authenticate caller
            identity and is not a fraud confirmation. Never automatically disconnect or block callers.
          </span>
        </div>
      </div>

      {errorMessage && (
        <div className="alert-error" role="alert" id="stream-error-alert" aria-live="assertive">
          <span className="alert-icon" aria-hidden="true">
            ✕
          </span>
          <p id="error-alert-text">{errorMessage}</p>
        </div>
      )}

      <main className="main-content">
        {/* ========================================================================= */}
        {/* TAB 1: Real-time Streaming Dashboard                                      */}
        {/* ========================================================================= */}
        {activeTab === 'stream' && (
          <div
            role="tabpanel"
            id="panel-stream"
            aria-labelledby="tab-stream-btn"
            tabIndex={0}
            className="tabpanel-content"
          >
            <h2 className="sr-only">Live Microphone Stream Analysis</h2>
            {/* Action Directive on High Anomaly — ONLY rendered when reliable and permitted */}
            {status === 'high' && liveOriginVerdictAllowed && (
              <section
                className="action-directive-high"
                role="alert"
                aria-live="assertive"
                id="high-status-action-banner"
              >
                <div className="action-icon" aria-hidden="true">
                  ⚠️
                </div>
                <div className="action-body">
                  <h4>ACTION DIRECTIVE</h4>
                  <p className="action-strong">
                    Pause and verify through a known independent channel.
                  </p>
                  <p className="action-subtext">
                    Elevated synthetic speech anomaly score sustained over multiple consecutive windows. Do
                    not rely on voice identity.
                  </p>
                </div>
              </section>
            )}

            {/* Main Grid Dashboard */}
            <section className="grid-dashboard" aria-label="Streaming Assessment Controls and Status">
              {/* Status Card */}
              <div className="card status-card">
                <h3 className="card-title">
                  <span>Assessment Status</span>
                  <span className="sub-label">4.0s Window • 1.0s Hop</span>
                </h3>

                <div
                  className={`status-indicator-box status-box-${status}`}
                  role="status"
                  aria-live="polite"
                  id="current-status-box"
                >
                  <div className="status-header-row">
                    {/* Accessible Non-Color Indicator Text Badge */}
                    <div className="non-color-indicator" id="accessible-status-indicator">
                      {status === 'low' && '[✓ LOW ANOMALY]'}
                      {status === 'uncertain' && '[? UNCERTAIN]'}
                      {status === 'high' && '[! HIGH ANOMALY]'}
                      {status === 'insufficient_evidence' && '[... INSUFFICIENT SPEECH]'}
                      {status === 'limited_reliability' && '[~ LIMITED RELIABILITY]'}
                      {status === 'error' && '[✕ ERROR]'}
                    </div>

                    <div
                      className={`verdict-permission-pill permission-${
                        liveOriginVerdictAllowed ? 'permitted' : 'withheld'
                      }`}
                      id="stream-verdict-permission-status"
                    >
                      <span className="permission-icon" aria-hidden="true">
                        {liveOriginVerdictAllowed ? '✓' : '⊘'}
                      </span>
                      <span>
                        Origin Verdict: {liveOriginVerdictAllowed ? 'Permitted' : 'Withheld (Gate Active)'}
                      </span>
                    </div>
                  </div>

                  <div className="status-label-heading" id="status-heading">
                    {status === 'low' && 'Low acoustic anomaly score — does not authenticate speech'}
                    {status === 'uncertain' && 'Uncertain / Ambiguous Zone'}
                    {status === 'high' && 'High Acoustic Anomaly Score'}
                    {status === 'insufficient_evidence' && 'Insufficient Active Speech'}
                    {status === 'limited_reliability' && 'Limited Reliability'}
                    {status === 'error' && 'Assessment Error'}
                  </div>

                  <p className="status-reason" id="status-reason-details">
                    {status === 'limited_reliability' && 'LIMITED RELIABILITY — ORIGIN VERDICT WITHHELD'}
                    {status === 'insufficient_evidence' && 'INSUFFICIENT EVIDENCE — ORIGIN VERDICT WITHHELD'}
                    {status !== 'limited_reliability' && status !== 'insufficient_evidence' && reasonDetails}
                  </p>
                  {(status === 'limited_reliability' || status === 'insufficient_evidence') && reasonDetails && (
                    <div className="status-reason-subtext" style={{ fontSize: '0.85rem', marginTop: '0.25rem' }}>
                      {reasonDetails}
                    </div>
                  )}
                </div>

                {/* Raw and temporally smoothed score display */}
                <div className="score-container" id="score-container">
                  {uncalibratedScore !== null ? (
                    <div className="score-block">
                      <span className="score-type" id="raw-score-label">{streamScoreLabel}:</span>
                      <span className="score-value" id="raw-score-value">
                        score {uncalibratedScore.toFixed(3)}
                      </span>
                      <span className="score-source-note" id="threshold-provenance-note" style={{ fontSize: '0.8rem' }}>
                        Low threshold: {thresholdLow.toFixed(3)} | High threshold: {thresholdHigh.toFixed(3)} ({operatingThresholdSource === 'unvalidated_heuristic' ? 'unvalidated heuristic' : operatingThresholdSource}). The intermediate range is UNCERTAIN; thresholds apply to the smoothed score/history.
                      </span>
                    </div>
                  ) : (
                    <div className="score-block">
                      <span className="score-type">Score:</span>
                      <span className="score-value-pending" id="score-pending-value">
                        {status === 'limited_reliability' || status === 'insufficient_evidence'
                          ? 'N/A — inference withheld'
                          : 'Awaiting 4.0s window...'}
                      </span>
                    </div>
                  )}
                  <div className="score-block">
                    <span className="score-type">EWMA-smoothed policy score:</span>
                    <span className="score-value" id="smoothed-score-value">
                      {smoothedScore !== null ? smoothedScore.toFixed(3) : 'N/A'}
                    </span>
                    <span className="score-source-note" style={{ fontSize: '0.8rem' }}>
                      Temporal decisions and explanations use the smoothed policy score/history, not necessarily the latest raw score.
                    </span>
                  </div>
                </div>

                {/* Control Actions (Require User Gesture) */}
                <div className="controls-row">
                  {!isRecording ? (
                    <button
                      type="button"
                      className="btn btn-primary"
                      onClick={startStream}
                      id="start-mic-button"
                    >
                      🎤 Start Microphone Stream
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="btn btn-danger"
                      onClick={stopAudioAndDisconnect}
                      disabled={recordingState === 'stopping'}
                      id="stop-mic-button"
                    >
                      {recordingState === 'stopping' ? '⏳ Stopping Stream...' : '⏹ Stop Stream & Disconnect'}
                    </button>
                  )}
                </div>
              </div>

              {/* Quality & Gating Metrics Card */}
              <div className="card metrics-card">
                <h3 className="card-title">
                  <span>Audio Quality & Speech Gating</span>
                </h3>

                <div className="metrics-list">
                  <div className="metric-row">
                    <span className="metric-name">Active Speech in Window</span>
                    <span
                      className={`metric-status ${
                        activeSpeechSeconds >= 0.8 ? 'metric-clean' : 'metric-warn'
                      }`}
                      id="speech-duration-status"
                    >
                      {activeSpeechSeconds.toFixed(2)}s / 4.0s{' '}
                      {activeSpeechSeconds >= 0.8 ? '(Sufficient)' : '(Insufficient < 0.8s)'}
                    </span>
                  </div>

                  <div className="metric-row">
                    <span className="metric-name">Clipping Distortion</span>
                    <span
                      className={`metric-status ${
                        qualityMetrics.isClipping ? 'metric-warn' : 'metric-clean'
                      }`}
                      id="clipping-status"
                    >
                      {qualityMetrics.isClipping
                        ? `Clipping Detected (${qualityMetrics.clippingCount} frame(s))`
                        : 'Clean'}
                    </span>
                  </div>

                  <div className="metric-row">
                    <span className="metric-name">Signal Energy (Silence Gate)</span>
                    <span
                      className={`metric-status ${
                        qualityMetrics.isSilent ? 'metric-warn' : 'metric-clean'
                      }`}
                      id="silence-status"
                    >
                      {qualityMetrics.isSilent ? 'Near Silence (RMS < 1e-4)' : 'Active Signal'}
                    </span>
                  </div>

                  <div className="metric-row">
                    <span className="metric-name">Timing Continuity</span>
                    <span
                      className={`metric-status ${
                        qualityMetrics.hasTimingGap ? 'metric-warn' : 'metric-clean'
                      }`}
                      id="timing-gap-status"
                    >
                      {qualityMetrics.hasTimingGap
                        ? `Discontinuity Flagged (${qualityMetrics.gapCount} gap(s))`
                        : 'Continuous'}
                    </span>
                  </div>
                </div>
              </div>
            </section>

          {/* Compact Stream Reliability Summary Section */}
          <section className="reliability-summary-card" data-testid="stream-reliability-summary" aria-label="Stream reliability summary">
            <div className="summary-title">Input Reliability Summary</div>
            <div className="summary-grid">
              <div className="summary-item">
                <span className="summary-label">Total Windows:</span>
                <span className="summary-value" id="stream-total-windows">{sessionCounters.total}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Reliable Windows:</span>
                <span className="summary-value" id="stream-reliable-windows">{streamReliableCount}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Limited Reliability:</span>
                <span className="summary-value" id="stream-limited-windows">{streamLimitedCount}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Insufficient Windows:</span>
                <span className="summary-value" id="stream-insufficient-windows">{streamInsufficientCount}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Degraded Quality (Diagnostic):</span>
                <span className="summary-value" id="stream-degraded-windows">{streamDegradedCount}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Origin Verdict Permitted:</span>
                <span className="summary-value" id="stream-verdict-permitted">
                  {streamOriginAllowed ? 'Yes (Permitted)' : 'No (Withheld)'}
                </span>
              </div>
            </div>
          </section>

          {/* Rolling Timeline Section */}
          <section className="timeline-section" aria-label="Rolling timeline of window evaluations">
            <h3 className="timeline-title">Rolling Window Assessment Timeline</h3>
            {timeline.length === 0 ? (
              <p className="timeline-empty" id="timeline-empty-message">
                No inference windows evaluated yet. Stream active speech for at least 4.0 seconds.
              </p>
            ) : (
              <div className="timeline-list" id="timeline-list" role="feed">
                {timeline.map((item) => (
                  <article
                    key={item.windowIndex}
                    className={`timeline-card timeline-status-${item.status}`}
                  >
                    <div className="timeline-header">
                      <span className="timeline-idx">Window #{item.windowIndex}</span>
                      <span className="timeline-range">
                        {item.windowStartSec.toFixed(1)}s - {item.windowEndSec.toFixed(1)}s
                      </span>
                      <span className={`timeline-badge badge-${item.status}`}>
                        {item.status.toUpperCase()}
                      </span>
                    </div>
                    <div className="timeline-body">
                      <span className="timeline-speech">
                        Speech: {item.speechSeconds.toFixed(2)}s
                      </span>
                      {item.syntheticScore !== null && (
                        <span className="timeline-score">
                          Raw: {item.syntheticScore.toFixed(3)}
                        </span>
                      )}
                      <span className="timeline-score">
                        Smoothed: {item.smoothedScore !== null ? item.smoothedScore.toFixed(3) : 'N/A'}
                      </span>
                      <span className={`badge-reliability badge-${item.inputReliability?.state || 'unknown'}`}>
                        Reliability: {(item.inputReliability?.state || 'unknown').toUpperCase()}
                      </span>
                      {(() => {
                        const qVal = typeof item.qualityStatus === 'object' ? item.qualityStatus.quality_status : String(item.qualityStatus);
                        return (
                          <span className={`timeline-quality-status badge-quality status-${qVal}`}>
                            {qVal.toUpperCase()}
                          </span>
                        );
                      })()}
                      <span className="timeline-inference">
                        Inference: {item.inferencePerformed ? 'Performed' : 'Withheld'}
                      </span>
                    </div>
                    {item.inputReliability?.reasons && item.inputReliability.reasons.length > 0 && (
                      <div className="timeline-reasons" style={{ fontSize: '0.8rem', marginTop: '0.25rem' }}>
                        Reasons: {item.inputReliability.reasons.join(', ')}
                      </div>
                    )}
                    <p className="timeline-reason">{item.reasonDetails}</p>
                  </article>
                ))}
              </div>
            )}
          </section>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: File Audio Analysis                                                */}
      {/* ========================================================================= */}
      {activeTab === 'file' && (
        <div
          role="tabpanel"
          id="panel-file"
          aria-labelledby="tab-file-btn"
          tabIndex={0}
          className="tabpanel-content"
        >
          <h2 className="sr-only">Offline Audio File Analysis</h2>
          <section className="file-analysis-card" aria-label="Offline audio file analysis">
          <form onSubmit={handleFileAnalysis}>
            <div
              className="file-dropzone"
              onClick={() => document.getElementById('audio-file-input')?.click()}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  document.getElementById('audio-file-input')?.click();
                }
              }}
              tabIndex={0}
              role="button"
              aria-label="Upload audio file"
            >
              <div className="file-dropzone-icon" aria-hidden="true">
                📁
              </div>
              <p className="file-dropzone-text">Click or select a WAV or FLAC file to analyze</p>
              <p className="file-dropzone-hint">
                Standard 16 kHz mono conversion and 4.0s windowing are applied in memory.
              </p>
              <input
                id="audio-file-input"
                type="file"
                accept=".wav,.flac,audio/wav,audio/flac"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setSelectedFile(e.target.files[0]);
                  }
                }}
              />
              {selectedFile && (
                <div className="file-selected-badge" id="selected-file-badge">
                  Selected: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
                </div>
              )}
            </div>

            <div className="file-form-grid">
              <div className="form-group">
                <label htmlFor="hop-seconds-input">Window Hop Step (seconds):</label>
                <input
                  id="hop-seconds-input"
                  type="number"
                  min="0.25"
                  max="4.0"
                  step="0.25"
                  value={fileHopSeconds}
                  onChange={(e) => setFileHopSeconds(parseFloat(e.target.value) || 1.0)}
                />
              </div>

              <div className="form-group">
                <label htmlFor="threshold-input">Decision Threshold:</label>
                <input
                  id="threshold-input"
                  type="number"
                  min="0.05"
                  max="0.95"
                  step="0.05"
                  value={fileThreshold}
                  onChange={(e) => setFileThreshold(parseFloat(e.target.value) || 0.65)}
                />
              </div>
            </div>

            <div className="controls-row">
              <button
                type="submit"
                className="btn btn-primary"
                disabled={!selectedFile || isFileAnalyzing}
                id="submit-file-analysis-button"
              >
                {isFileAnalyzing ? '⏳ Evaluating 4.0s Windows...' : '⚡ Analyze Audio File'}
              </button>
            </div>
          </form>

          {fileError && (
            <div className="alert-error" role="alert" style={{ marginTop: '1.25rem' }}>
              <span className="alert-icon" aria-hidden="true">
                ✕
              </span>
              <p id="file-error-text">{fileError}</p>
            </div>
          )}

          {fileResult && (
            <div className="file-results-card" style={{ marginTop: '1.5rem' }}>
              <div className="file-verdict-summary">
                <div className="verdict-stat">
                  <div className="verdict-stat-label">Advisory Verdict</div>
                  {(() => {
                    if (fileVerdictPermitted) {
                      const dec = fileResult.final_decision_state?.toLowerCase();
                      if (dec === 'high') {
                        return (
                          <div
                            className="verdict-stat-value"
                            id="file-decision-heading"
                            style={{ color: 'var(--accent-red)' }}
                          >
                            [! HIGH ANOMALY]
                          </div>
                        );
                      }
                      if (dec === 'uncertain') {
                        return (
                          <div
                            className="verdict-stat-value"
                            id="file-decision-heading"
                            style={{ color: 'var(--accent-amber)' }}
                          >
                            [~ MIXED / UNCERTAIN]
                          </div>
                        );
                      }
                      if (dec === 'low') {
                        return (
                          <div
                            className="verdict-stat-value"
                            id="file-decision-heading"
                            style={{ color: 'var(--accent-emerald)' }}
                          >
                            [✓ LOW ANOMALY]
                          </div>
                        );
                      }
                    }

                    let headingText = '[✕ CONTRACT ERROR]';
                    let subtitleText = 'CONTRACT ERROR — ORIGIN VERDICT WITHHELD';
                    let color = 'var(--accent-red)';

                    if (fileResult.status === 'limited_reliability') {
                      headingText = '[~ LIMITED RELIABILITY]';
                      subtitleText = 'LIMITED RELIABILITY — ORIGIN VERDICT WITHHELD';
                      color = 'var(--accent-amber)';
                    } else if (fileResult.status === 'insufficient_evidence') {
                      headingText = '[... INSUFFICIENT SPEECH]';
                      subtitleText = 'INSUFFICIENT EVIDENCE — ORIGIN VERDICT WITHHELD';
                      color = 'var(--accent-amber)';
                    } else if (fileResult.status === 'error') {
                      headingText = '[✕ ERROR]';
                      subtitleText = 'ANALYSIS ERROR — ORIGIN VERDICT WITHHELD';
                      color = 'var(--accent-red)';
                    } else if (fileResult.status === 'unsupported') {
                      headingText = '[? UNSUPPORTED]';
                      subtitleText = 'UNSUPPORTED RESULT — ORIGIN VERDICT WITHHELD';
                      color = 'var(--accent-amber)';
                    }

                    return (
                      <>
                        <div
                          className="verdict-stat-value"
                          id="file-decision-heading"
                          style={{ color }}
                        >
                          {headingText}
                        </div>
                        <div
                          className="verdict-subtitle"
                          style={{ fontSize: '0.85rem' }}
                        >
                          {subtitleText}
                        </div>
                      </>
                    );
                  })()}
                  <div
                    className={`verdict-permission-pill permission-${
                      fileVerdictPermitted ? 'permitted' : 'withheld'
                    }`}
                    id="file-verdict-permission-pill"
                    style={{ marginTop: '0.5rem' }}
                  >
                    <span className="permission-icon" aria-hidden="true">
                      {fileVerdictPermitted ? '✓' : '⊘'}
                    </span>
                    <span>
                      Origin Verdict: {fileVerdictPermitted ? 'Permitted' : 'Withheld (Gate Active)'}
                    </span>
                  </div>
                </div>

                <div className="verdict-stat">
                  <div className="verdict-stat-label">{fileMeanScoreLabel}</div>
                  <div className="verdict-stat-value" id="file-mean-score">
                    {fileResult.mean_synthetic_score !== null
                      ? fileResult.mean_synthetic_score.toFixed(3)
                      : 'N/A — inference not performed'}
                  </div>
                </div>

                <div className="verdict-stat">
                  <div className="verdict-stat-label">{filePeakScoreLabel}</div>
                  <div className="verdict-stat-value" id="file-peak-score">
                    {fileResult.max_synthetic_score !== null
                      ? fileResult.max_synthetic_score.toFixed(3)
                      : 'N/A — inference not performed'}
                  </div>
                </div>

                <div className="verdict-stat">
                  <div className="verdict-stat-label">Duration / Windows</div>
                  <div className="verdict-stat-value">
                    {fileResult.duration_seconds.toFixed(1)}s ({fileResult.total_windows} win)
                  </div>
                </div>
              </div>

              <div
                className="threshold-provenance-banner"
                style={{
                  margin: '1rem 0',
                  padding: '0.75rem 1rem',
                  borderRadius: '6px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  fontSize: '0.85rem',
                }}
              >
                <strong style={{ color: 'var(--text-primary)' }}>Threshold Provenance: </strong>
                <span>
                  Effective threshold: {fileResult.effective_threshold.toFixed(3)} ({fileResult.operating_threshold_source === 'unvalidated_heuristic' ? 'unvalidated heuristic' : fileResult.operating_threshold_source}). Uncalibrated model output; does not represent calibrated probability.
                </span>
              </div>

              {/* Compact File Input Reliability Summary */}
              <section className="reliability-summary-card" data-testid="file-reliability-summary" aria-label="File reliability summary">
                <div className="summary-title">Input Reliability Summary</div>
                <div className="summary-grid">
                  <div className="summary-item">
                    <span className="summary-label">Reliable Windows:</span>
                    <span className="summary-value" id="file-reliable-windows">{fileResult.reliable_window_count}</span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Limited Reliability:</span>
                    <span className="summary-value" id="file-limited-windows">{fileResult.limited_reliability_window_count}</span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Insufficient Windows:</span>
                    <span className="summary-value" id="file-insufficient-windows">{fileResult.insufficient_reliability_window_count}</span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Degraded Quality (Diagnostic):</span>
                    <span className="summary-value" id="file-degraded-windows">{fileResult.degraded_quality_window_count}</span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Origin Verdict Permitted:</span>
                    <span className="summary-value" id="file-verdict-permitted">
                      {fileResult.quality_origin_verdict_allowed ? 'Yes (Permitted)' : 'No (Withheld)'}
                    </span>
                  </div>
                </div>
              </section>

              {/* Action Directive on High Anomaly File — permitted ONLY when success, origin permitted, reliable windows > 0, and no degraded/limited windows */}
              {fileVerdictPermitted &&
                fileResult.final_decision_state?.toLowerCase() === 'high' && (
                <section
                  className="action-directive-high"
                  role="alert"
                  aria-live="assertive"
                  style={{ marginBottom: '1.25rem' }}
                >
                  <div className="action-icon" aria-hidden="true">
                    ⚠️
                  </div>
                  <div className="action-body">
                    <h4>ACTION DIRECTIVE</h4>
                    <p className="action-strong">
                      Pause and verify through a known independent channel.
                    </p>
                    <p className="action-subtext">
                      Multiple consecutive temporal windows exhibited elevated synthetic acoustic anomaly scores.
                    </p>
                  </div>
                </section>
              )}

              <p className="status-reason" style={{ marginBottom: '1rem' }}>
                {fileResult.temporal_rule_explanation}
              </p>

              {/* Per-window breakdown table */}
              <div className="window-table-container">
                <table className="window-table" aria-label="Per-window acoustic evaluation records">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Time Range</th>
                      <th>Reliability</th>
                      <th>Quality Status</th>
                      <th>Inference</th>
                      <th>Bona Fide Score</th>
                      <th>Raw Uncalibrated Score</th>
                      <th>Smoothed Policy Score</th>
                      <th>Active Speech</th>
                      <th>Temporal State</th>
                    </tr>
                  </thead>
                  <tbody>
                    {fileResult.windows.map((win) => {
                      const state = fileWindowState(win);
                      const isWithheld = win.is_eligible === false || win.input_reliability?.origin_verdict_allowed === false;
                      const relState = win.input_reliability?.state ?? (win.is_eligible ? 'reliable' : 'insufficient');
                      const qualityVal = typeof win.quality_status === 'object' ? win.quality_status.quality_status : String(win.quality_status);
                      return (
                        <tr key={win.window_index}>
                          <td>#{win.window_index}</td>
                          <td>
                            {win.start_seconds.toFixed(1)}s - {win.end_seconds.toFixed(1)}s
                          </td>
                          <td>
                            <span className={`badge-reliability badge-${relState}`}>
                              {relState.toUpperCase()}
                            </span>
                            {win.input_reliability?.reasons && win.input_reliability.reasons.length > 0 && (
                              <div className="window-reliability-reasons" style={{ fontSize: '0.75rem' }}>
                                {win.input_reliability.reasons.join(', ')}
                              </div>
                            )}
                          </td>
                          <td>
                            <span className={`badge-quality status-${qualityVal}`}>
                              {qualityVal.toUpperCase()}
                            </span>
                          </td>
                          <td>
                            <span className={`badge-inference ${isWithheld ? 'badge-withheld' : 'badge-performed'}`}>
                              {isWithheld ? 'WITHHELD' : 'PERFORMED'}
                            </span>
                          </td>
                          <td>
                            {win.raw_bonafide_score !== null && win.raw_bonafide_score !== undefined && !isWithheld
                              ? win.raw_bonafide_score.toFixed(3)
                              : 'N/A'}
                          </td>
                          <td>
                            {win.synthetic_score !== null && win.synthetic_score !== undefined && !isWithheld
                              ? win.synthetic_score.toFixed(3)
                              : 'N/A'}
                          </td>
                          <td>
                            {win.is_eligible !== false && win.smoothed_score !== null && win.smoothed_score !== undefined && !isWithheld
                              ? win.smoothed_score.toFixed(3)
                              : 'N/A'}
                          </td>
                          <td>{win.active_speech_seconds.toFixed(2)}s</td>
                          <td>
                            <span className={`timeline-badge ${state.className}`}>{state.label}</span>
                            {win.decision_reason_details && (
                              <div className="timeline-reason">{win.decision_reason_details}</div>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
        </div>
      )}
      </main>
    </div>
  );
};

export default App;
