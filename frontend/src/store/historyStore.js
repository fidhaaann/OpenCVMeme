import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const MAX_HISTORY_ITEMS = 50;

const useHistoryStore = create(
  persist(
    (set, get) => ({
      history: [],
      
      /**
       * Add a detection result to history
       */
      addToHistory: (result) => {
        const historyItem = {
          id: result.id || Date.now().toString(),
          prediction: result.prediction,
          confidence: result.confidence,
          isConfident: result.is_confident || result.isConfident,
          confidenceBreakdown: result.confidence_breakdown || result.confidenceBreakdown,
          timestamp: result.timestamp || new Date().toISOString(),
          imagePreview: result.imagePreview || null,
          gifUrl: result.gif_url || result.gifUrl || null,
          processingTime: result.processing_time_ms || result.processingTime || null,
          detectionStatus: result.detection_status || result.detectionStatus || null,
        };
        
        set((state) => ({
          history: [historyItem, ...state.history].slice(0, MAX_HISTORY_ITEMS),
        }));
        
        return historyItem;
      },
      
      /**
       * Remove an item from history
       */
      removeFromHistory: (id) => {
        set((state) => ({
          history: state.history.filter((item) => item.id !== id),
        }));
      },
      
      /**
       * Clear all history
       */
      clearHistory: () => {
        set({ history: [] });
      },
      
      /**
       * Get history statistics
       */
      getStats: () => {
        const history = get().history;
        
        if (history.length === 0) {
          return {
            totalDetections: 0,
            confidentDetections: 0,
            averageConfidence: 0,
            memeDistribution: {},
          };
        }
        
        const confidentDetections = history.filter((item) => item.isConfident);
        const averageConfidence = history.reduce((sum, item) => sum + item.confidence, 0) / history.length;
        
        const memeDistribution = history.reduce((acc, item) => {
          acc[item.prediction] = (acc[item.prediction] || 0) + 1;
          return acc;
        }, {});
        
        return {
          totalDetections: history.length,
          confidentDetections: confidentDetections.length,
          averageConfidence,
          memeDistribution,
        };
      },
      
      /**
       * Get most recent confident detection
       */
      getLastConfident: () => {
        const history = get().history;
        return history.find((item) => item.isConfident) || null;
      },
    }),
    {
      name: 'meme-detector-history',
      partialize: (state) => ({
        // Only persist history without image previews (to save space)
        history: state.history.map((item) => ({
          ...item,
          imagePreview: null, // Don't persist large base64 images
        })),
      }),
    }
  )
);

export default useHistoryStore;
