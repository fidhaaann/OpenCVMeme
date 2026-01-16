'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Camera, 
  CameraOff, 
  Sparkles, 
  Zap, 
  Hand, 
  User,
  Activity,
  Github
} from 'lucide-react'

// Types
interface DetectionResult {
  prediction: string
  confidence: number
  is_confident: boolean
  detection_status: {
    face: boolean
    left_hand: boolean
    right_hand: boolean
  }
  gif: string | null
  error?: string
}

// Gesture hints data
const GESTURE_HINTS = [
  { name: 'Cooked', emoji: '🙏', description: 'Prayer hands' },
  { name: 'DiCaprio', emoji: '👏', description: 'Clapping' },
  { name: 'Think', emoji: '🤔', description: 'Thinking pose' },
  { name: 'Vanish', emoji: '✌️', description: 'Peace sign' },
  { name: 'Speed', emoji: '💨', description: 'Fast movement' },
]

// API Configuration
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws'

export default function Home() {
  // State
  const [isStreaming, setIsStreaming] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [detection, setDetection] = useState<DetectionResult | null>(null)
  const [fps, setFps] = useState(0)
  const [error, setError] = useState<string | null>(null)
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const frameIdRef = useRef<number>(0)
  const lastTimeRef = useRef<number>(Date.now())
  const fpsCountRef = useRef<number>(0)

  // Start camera and WebSocket
  const startStream = useCallback(async () => {
    try {
      setError(null)
      
      // Get camera stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 1280 }, 
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false
      })
      
      streamRef.current = stream
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      
      // Connect WebSocket
      const ws = new WebSocket(WS_URL)
      
      ws.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        setIsStreaming(true)
        startProcessing()
      }
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data) as DetectionResult
        setDetection(data)
        
        // Calculate FPS
        fpsCountRef.current++
        const now = Date.now()
        if (now - lastTimeRef.current >= 1000) {
          setFps(fpsCountRef.current)
          fpsCountRef.current = 0
          lastTimeRef.current = now
        }
      }
      
      ws.onerror = (e) => {
        console.error('WebSocket error:', e)
        setError('Connection error. Make sure the backend is running.')
      }
      
      ws.onclose = () => {
        console.log('WebSocket closed')
        setIsConnected(false)
      }
      
      wsRef.current = ws
      
    } catch (err) {
      console.error('Error starting stream:', err)
      setError('Failed to access camera. Please allow camera permissions.')
    }
  }, [])

  // Stop stream
  const stopStream = useCallback(() => {
    // Stop frame processing
    if (frameIdRef.current) {
      cancelAnimationFrame(frameIdRef.current)
      frameIdRef.current = 0
    }
    
    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    
    // Stop camera
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track: MediaStreamTrack) => track.stop())
      streamRef.current = null
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    
    setIsStreaming(false)
    setIsConnected(false)
    setDetection(null)
    setFps(0)
  }, [])

  // Process frames
  const startProcessing = useCallback(() => {
    const processFrame = () => {
      if (!videoRef.current || !canvasRef.current || !wsRef.current) return
      if (wsRef.current.readyState !== WebSocket.OPEN) return
      
      const video = videoRef.current
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      
      if (!ctx || !video.videoWidth) {
        frameIdRef.current = requestAnimationFrame(processFrame)
        return
      }
      
      // Set canvas size
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      
      // Draw frame WITHOUT mirroring for the model (model expects non-mirrored)
      ctx.drawImage(video, 0, 0)
      
      // Send frame to server at higher quality for better detection
      const imageData = canvas.toDataURL('image/jpeg', 0.8)
      wsRef.current.send(JSON.stringify({ image: imageData }))
      
      // Schedule next frame (~10fps for good balance of accuracy and performance)
      setTimeout(() => {
        frameIdRef.current = requestAnimationFrame(processFrame)
      }, 100)
    }
    
    frameIdRef.current = requestAnimationFrame(processFrame)
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStream()
    }
  }, [stopStream])

  return (
    <main className="min-h-screen bg-dark">
      {/* Header */}
      <motion.header 
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className="glass sticky top-0 z-50 px-6 py-4"
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <motion.div 
            className="flex items-center gap-3"
            whileHover={{ scale: 1.02 }}
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-primary flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold gradient-text">Meme Detector</h1>
          </motion.div>
          
          <div className="flex items-center gap-6">
            {/* FPS Counter */}
            {isStreaming && (
              <motion.div 
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-2 text-sm"
              >
                <Activity className="w-4 h-4 text-accent-yellow" />
                <span className="font-mono text-accent-yellow">{fps} FPS</span>
              </motion.div>
            )}
            
            {/* Detection indicators */}
            {isStreaming && detection && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center gap-2"
              >
                <StatusIndicator 
                  active={detection.detection_status?.face} 
                  icon={<User className="w-3 h-3" />}
                  label="Face"
                />
                <StatusIndicator 
                  active={detection.detection_status?.left_hand} 
                  icon={<Hand className="w-3 h-3" />}
                  label="L"
                />
                <StatusIndicator 
                  active={detection.detection_status?.right_hand} 
                  icon={<Hand className="w-3 h-3 scale-x-[-1]" />}
                  label="R"
                />
              </motion.div>
            )}
            
            {/* Connection status */}
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 live-indicator' : 'bg-gray-600'}`} />
              <span className="text-sm text-gray-400">
                {isConnected ? 'Connected' : 'Offline'}
              </span>
            </div>
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Hero Section (shown when not streaming) */}
        <AnimatePresence>
          {!isStreaming && (
            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="text-center mb-12"
            >
              <motion.h2 
                className="text-5xl md:text-6xl font-bold mb-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <span className="gradient-text">AI-Powered</span>
                <br />
                <span className="text-white">Meme Generator</span>
              </motion.h2>
              <motion.p 
                className="text-gray-400 text-lg max-w-2xl mx-auto mb-8"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
              >
                Express yourself through gestures. Our AI detects your face and hands 
                in real-time to generate matching memes instantly.
              </motion.p>
              
              {/* Gesture hints */}
              <motion.div 
                className="flex flex-wrap justify-center gap-3 mb-8"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
              >
                {GESTURE_HINTS.map((hint, i) => (
                  <motion.div
                    key={hint.name}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.7 + i * 0.1 }}
                    className="glass px-4 py-2 rounded-full flex items-center gap-2 hover:border-primary transition-colors"
                  >
                    <span className="text-xl">{hint.emoji}</span>
                    <span className="text-sm text-gray-300">{hint.name}</span>
                  </motion.div>
                ))}
              </motion.div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Error message */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-center"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main Grid */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Video Feed */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="relative"
          >
            <div className="video-container aspect-video glow-border rounded-2xl overflow-hidden bg-dark-card">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              <canvas ref={canvasRef} className="hidden" />
              
              {/* Overlay - Prediction */}
              {isStreaming && detection && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="absolute top-0 left-0 right-0 p-4 bg-gradient-to-b from-black/70 to-transparent"
                >
                  <div className={`text-2xl font-bold ${detection.is_confident ? 'text-accent-yellow' : 'text-gray-500'}`}>
                    {detection.prediction?.toUpperCase() || 'NONE'}
                  </div>
                  <div className="text-sm text-gray-400">
                    Confidence: {((detection.confidence || 0) * 100).toFixed(1)}%
                  </div>
                </motion.div>
              )}
              
              {/* Empty state */}
              {!isStreaming && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-dark-card">
                  <motion.div
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                    className="w-20 h-20 rounded-full bg-gradient-primary flex items-center justify-center mb-4"
                  >
                    <Camera className="w-10 h-10 text-white" />
                  </motion.div>
                  <p className="text-gray-400">Click Start to begin</p>
                </div>
              )}
            </div>
            
            {/* Control Button */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={isStreaming ? stopStream : startStream}
              className={`w-full mt-4 py-4 rounded-xl font-semibold text-lg transition-all duration-300 flex items-center justify-center gap-3
                ${isStreaming 
                  ? 'bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30' 
                  : 'bg-gradient-primary text-white shadow-glow hover:shadow-glow-pink'
                }`}
            >
              {isStreaming ? (
                <>
                  <CameraOff className="w-5 h-5" />
                  Stop Detection
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  Start Detection
                </>
              )}
            </motion.button>
          </motion.div>

          {/* Meme Output */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="glass rounded-2xl p-6 flex flex-col items-center justify-center min-h-[400px]"
          >
            <AnimatePresence mode="wait">
              {detection?.is_confident && detection.gif ? (
                <motion.div
                  key={detection.prediction}
                  initial={{ opacity: 0, scale: 0.8, rotateY: 90 }}
                  animate={{ opacity: 1, scale: 1, rotateY: 0 }}
                  exit={{ opacity: 0, scale: 0.8, rotateY: -90 }}
                  transition={{ duration: 0.4 }}
                  className="text-center"
                >
                  <div className="meme-card rounded-xl overflow-hidden shadow-glow mb-4">
                    <img
                      src={`${API_URL}/gif/${detection.gif}`}
                      alt={detection.prediction}
                      className="max-w-full max-h-[50vh] object-contain"
                    />
                  </div>
                  <motion.h3 
                    className="text-2xl font-bold gradient-text uppercase"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                  >
                    {detection.prediction}
                  </motion.h3>
                </motion.div>
              ) : (
                <motion.div
                  key="placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-center"
                >
                  <motion.div 
                    className="text-8xl mb-6"
                    animate={{ 
                      rotate: [0, 10, -10, 0],
                      scale: [1, 1.1, 1]
                    }}
                    transition={{ duration: 4, repeat: Infinity }}
                  >
                    🎭
                  </motion.div>
                  <h3 className="text-xl font-semibold text-gray-400 mb-2">
                    {isStreaming ? 'Make a gesture!' : 'Ready to detect'}
                  </h3>
                  <p className="text-gray-500 text-sm max-w-xs">
                    {isStreaming 
                      ? 'Try one of the gestures shown above' 
                      : 'Start the camera to begin detecting gestures'
                    }
                  </p>
                  
                  {/* Gesture grid when streaming but no detection */}
                  {isStreaming && (
                    <motion.div 
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className="grid grid-cols-3 gap-2 mt-6"
                    >
                      {GESTURE_HINTS.slice(0, 3).map((hint) => (
                        <div key={hint.name} className="p-3 rounded-lg bg-dark-lighter text-center">
                          <div className="text-2xl mb-1">{hint.emoji}</div>
                          <div className="text-xs text-gray-500">{hint.name}</div>
                        </div>
                      ))}
                    </motion.div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>

        {/* Instructions Section */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-12 glass rounded-2xl p-8"
        >
          <h3 className="text-2xl font-bold gradient-text mb-6 text-center">How It Works</h3>
          <div className="grid md:grid-cols-3 gap-6">
            <StepCard 
              number={1}
              title="Start Camera"
              description="Click the Start button to enable your webcam and connect to the AI"
              icon={<Camera className="w-6 h-6" />}
            />
            <StepCard 
              number={2}
              title="Make Gestures"
              description="Position yourself so your face and hands are visible, then strike a pose"
              icon={<Hand className="w-6 h-6" />}
            />
            <StepCard 
              number={3}
              title="Get Memes"
              description="Watch as the AI recognizes your gesture and displays the matching meme"
              icon={<Sparkles className="w-6 h-6" />}
            />
          </div>
        </motion.section>
      </div>

      {/* Footer */}
      <footer className="mt-12 border-t border-dark-border py-6">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between text-sm text-gray-500">
          <p>Built with Next.js, FastAPI & MediaPipe</p>
          <a 
            href="https://github.com" 
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 hover:text-primary transition-colors"
          >
            <Github className="w-4 h-4" />
            GitHub
          </a>
        </div>
      </footer>
    </main>
  )
}

// Status Indicator Component
function StatusIndicator({ 
  active, 
  icon, 
  label 
}: { 
  active?: boolean
  icon: React.ReactNode
  label: string 
}) {
  return (
    <motion.div
      animate={{ 
        backgroundColor: active ? 'rgba(153, 41, 234, 0.3)' : 'rgba(50, 50, 50, 0.5)',
        borderColor: active ? '#9929EA' : '#333'
      }}
      className="flex items-center gap-1 px-2 py-1 rounded-full border text-xs"
    >
      <span className={active ? 'text-primary' : 'text-gray-500'}>{icon}</span>
      <span className={active ? 'text-primary' : 'text-gray-500'}>{label}</span>
    </motion.div>
  )
}

// Step Card Component
function StepCard({ 
  number, 
  title, 
  description, 
  icon 
}: { 
  number: number
  title: string
  description: string
  icon: React.ReactNode 
}) {
  return (
    <motion.div 
      whileHover={{ y: -5 }}
      className="text-center p-6 rounded-xl bg-dark-lighter border border-dark-border hover:border-primary/50 transition-colors"
    >
      <div className="w-12 h-12 rounded-full bg-gradient-primary mx-auto mb-4 flex items-center justify-center text-white">
        {icon}
      </div>
      <div className="text-accent-yellow font-mono text-sm mb-2">Step {number}</div>
      <h4 className="text-lg font-semibold text-white mb-2">{title}</h4>
      <p className="text-gray-400 text-sm">{description}</p>
    </motion.div>
  )
}
