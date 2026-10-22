class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 1024;
    this.buffer = new Int16Array(this.bufferSize);
    this.bufferIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const channelData = input[0];
    for (let i = 0; i < channelData.length; i++) {
      // Clamp float32 sample to [-1.0, 1.0] and convert to 16-bit signed PCM
      const s = Math.max(-1.0, Math.min(1.0, channelData[i]));
      this.buffer[this.bufferIndex++] = s < 0 ? s * 0x8000 : s * 0x7fff;

      if (this.bufferIndex >= this.bufferSize) {
        const copy = this.buffer.slice();
        this.port.postMessage(copy.buffer, [copy.buffer]);
        this.bufferIndex = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-worklet-processor', PCMWorkletProcessor);
