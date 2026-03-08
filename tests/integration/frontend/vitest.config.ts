import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(new URL("../../../frontend/package.json", import.meta.url));

const reactModule = require("@vitejs/plugin-react");
const vitestConfig = require("vitest/config");
const react = reactModule.default ?? reactModule;
const defineConfig = vitestConfig.defineConfig ?? ((config: unknown) => config);

const frontendRoot = fileURLToPath(new URL("../../../frontend", import.meta.url));
const repoRoot = fileURLToPath(new URL("../../..", import.meta.url));
const setupFile = fileURLToPath(new URL("./vitest.setup.ts", import.meta.url));
const testingLibraryReactPath = fileURLToPath(
  new URL("../../../frontend/node_modules/@testing-library/react", import.meta.url)
);
const testingLibraryUserEventPath = fileURLToPath(
  new URL("../../../frontend/node_modules/@testing-library/user-event", import.meta.url)
);
const testingLibraryJestDomPath = fileURLToPath(
  new URL("../../../frontend/node_modules/@testing-library/jest-dom", import.meta.url)
);
const reactPath = fileURLToPath(new URL("../../../frontend/node_modules/react", import.meta.url));
const reactDomPath = fileURLToPath(new URL("../../../frontend/node_modules/react-dom", import.meta.url));

export default defineConfig({
  plugins: [react()],
  root: frontendRoot,
  resolve: {
    alias: {
      "@testing-library/react": testingLibraryReactPath,
      "@testing-library/user-event": testingLibraryUserEventPath,
      "@testing-library/jest-dom": testingLibraryJestDomPath,
      react: reactPath,
      "react-dom": reactDomPath,
    },
  },
  server: {
    fs: {
      allow: [repoRoot],
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: [setupFile],
    include: ["../tests/integration/frontend/**/*.{test,spec}.{ts,tsx}"],
  },
});
