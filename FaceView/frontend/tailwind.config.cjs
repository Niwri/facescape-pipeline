/********************
 * Tailwind Config  *
 ********************/
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}",
    "./src/components/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f5f9ff",
          100: "#e0ecff",
          500: "#2563eb",
          600: "#1d4ed8"
        }
      }
    }
  },
  plugins: []
};
