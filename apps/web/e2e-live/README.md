# Live browser contract

The live suite runs the built React application through Vite's HTTP/WebSocket
proxy against a mock-weight backend and uses synthetic browser audio. It
validates readiness, file, streaming, and graceful-stop transport/UI contracts;
it does not measure detector accuracy or Safari/production performance.
