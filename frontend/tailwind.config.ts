import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        leaf: {
          50: '#f4fbf5',
          100: '#e4f5e8',
          300: '#9ed9ab',
          500: '#3e9b59',
          700: '#23683a',
          900: '#183d26',
        },
        oat: '#f7f3ec',
        ink: '#172033',
        mint: '#e9f7ef',
        berry: '#9a3450',
      },
      boxShadow: {
        soft: '0 18px 50px rgba(32, 50, 40, 0.08)',
      },
    },
  },
  plugins: [],
} satisfies Config;
