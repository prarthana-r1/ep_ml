import { defineConfig } from 'vite'
 import react from '@vitejs/plugin-react'
 export default defineConfig({
 plugins: [react()],
 server: { 
    proxy: { 
        '/api': 'http://localhost:4000',
        target: 'http://127.0.0.1:5000',

  } }
 })