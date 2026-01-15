import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Custom color palette
        'primary': {
          DEFAULT: '#9929EA',
          dark: '#7A1FBF',
          light: '#B44EF7',
        },
        'accent': {
          pink: '#FF5FCF',
          yellow: '#FAEB92',
        },
        'dark': {
          DEFAULT: '#000000',
          lighter: '#0A0A0A',
          card: '#111111',
          border: '#1A1A1A',
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'gradient-primary': 'linear-gradient(135deg, #9929EA 0%, #FF5FCF 100%)',
        'gradient-glow': 'linear-gradient(135deg, rgba(153, 41, 234, 0.3) 0%, rgba(255, 95, 207, 0.3) 100%)',
      },
      boxShadow: {
        'glow': '0 0 30px rgba(153, 41, 234, 0.4)',
        'glow-pink': '0 0 30px rgba(255, 95, 207, 0.4)',
        'glow-yellow': '0 0 20px rgba(250, 235, 146, 0.3)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'float': 'float 6s ease-in-out infinite',
        'gradient': 'gradient 8s ease infinite',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 20px rgba(153, 41, 234, 0.4)' },
          '100%': { boxShadow: '0 0 40px rgba(255, 95, 207, 0.6)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        gradient: {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
      },
    },
  },
  plugins: [],
}

export default config
