import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 6001,
    host: '0.0.0.0'
  },
  build: {
    rollupOptions: {
      output: {
        // Тяжёлые vendor-либы — в отдельные чанки: кэшируются браузером между деплоями,
        // не инвалидируются при каждом изменении кода приложения. Порядок важен:
        // react-markdown/recharts проверяем до react (react-router тоже идёт в react).
        manualChunks(id) {
          if (!id.includes('node_modules')) return;
          if (id.includes('recharts')) return 'recharts';
          if (id.includes('react-markdown') || id.includes('remark')) return 'markdown';
          if (id.includes('react')) return 'react';
        },
      },
    },
  },
})