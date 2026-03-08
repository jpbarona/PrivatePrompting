import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { defineConfig, type Plugin } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

type BackendState = {
  status: 'idle' | 'starting' | 'ready' | 'error'
  apiUrl: string | null
  error: string | null
  logs: string[]
  process: ChildProcessWithoutNullStreams | null
}

type StartRequest = {
  hostIp?: string
  dhtPort?: number
  bootstrapMaddr?: string
  runId?: string
  apiPort?: number
  apiHost?: string
}

const backendState: BackendState = {
  status: 'idle',
  apiUrl: null,
  error: null,
  logs: [],
  process: null,
}

function appendLog(line: string) {
  if (!line.trim()) return
  backendState.logs.push(line)
  if (backendState.logs.length > 200) {
    backendState.logs = backendState.logs.slice(-200)
  }
}

function readJsonBody(req: NodeJS.ReadableStream): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    let raw = ''
    req.on('data', (chunk) => {
      raw += chunk
    })
    req.on('end', () => {
      if (!raw) {
        resolve({})
        return
      }
      try {
        resolve(JSON.parse(raw) as Record<string, unknown>)
      } catch (error) {
        reject(error)
      }
    })
    req.on('error', reject)
  })
}

function choosePython(repoRoot: string) {
  if (process.env.BACKEND_PYTHON?.trim()) return process.env.BACKEND_PYTHON.trim()
  const venvPython = path.join(repoRoot, '.venv', 'bin', 'python')
  if (fs.existsSync(venvPython)) return venvPython
  return 'python3'
}

function detectHostIp(): string {
  const interfaces = os.networkInterfaces()
  for (const addresses of Object.values(interfaces)) {
    if (!addresses) continue
    for (const address of addresses) {
      if (
        address.family === 'IPv4' &&
        !address.internal &&
        (address.address.startsWith('10.') ||
          address.address.startsWith('192.168.') ||
          /^172\.(1[6-9]|2\d|3[0-1])\./.test(address.address))
      ) {
        return address.address
      }
    }
  }
  return '127.0.0.1'
}

function backendLauncherPlugin(): Plugin {
  const repoRoot = path.resolve(__dirname, '..')
  const apiScript = path.join(repoRoot, 'inference', 'api.py')

  return {
    name: 'backend-launcher',
    configureServer(server) {
      server.httpServer?.once('close', () => {
        if (backendState.process && !backendState.process.killed) {
          backendState.process.kill()
        }
      })

      server.middlewares.use('/__backend/start', async (req, res, next) => {
        if (req.method !== 'POST') {
          next()
          return
        }

        try {
          const body = (await readJsonBody(req)) as StartRequest
          const hostIp = body.hostIp?.toString().trim() || detectHostIp()
          const dhtPort = Number(body.dhtPort ?? 43313)
          const bootstrapMaddr = body.bootstrapMaddr?.toString().trim() ?? ''
          const runId = body.runId?.toString().trim() ?? ''
          const apiPort = Number(body.apiPort ?? 8000)
          const apiHost = body.apiHost?.toString().trim() || '0.0.0.0'

          if (!bootstrapMaddr || !Number.isFinite(dhtPort) || dhtPort <= 0) {
            res.statusCode = 400
            res.setHeader('Content-Type', 'application/json')
            res.end(
              JSON.stringify({
                error: 'bootstrapMaddr is required',
              }),
            )
            return
          }

          if (
            backendState.process &&
            !backendState.process.killed &&
            (backendState.status === 'starting' || backendState.status === 'ready')
          ) {
            res.statusCode = 200
            res.setHeader('Content-Type', 'application/json')
            res.end(
              JSON.stringify({
                status: backendState.status,
                apiUrl: backendState.apiUrl,
              }),
            )
            return
          }

          backendState.status = 'starting'
          backendState.error = null
          backendState.logs = []
          backendState.apiUrl = `http://localhost:${apiPort}`

          const pythonBin = choosePython(repoRoot)
          const args = [
            apiScript,
            '--host-ip',
            hostIp,
            '--dht-port',
            String(dhtPort),
            '--bootstrap-maddr',
            bootstrapMaddr,
            '--api-port',
            String(apiPort),
            '--api-host',
            apiHost,
          ]
          if (runId) {
            args.push('--run-id', runId)
          }

          const child = spawn(pythonBin, args, {
            cwd: repoRoot,
            env: { ...process.env, PYTHONUNBUFFERED: '1' },
          })
          backendState.process = child

          child.stdout.on('data', (chunk: Buffer) => {
            const lines = chunk.toString().split('\n')
            for (const line of lines) {
              appendLog(line)
              if (
                backendState.status === 'starting' &&
                (line.includes('[api] Ready.') || line.includes('Application startup complete.'))
              ) {
                backendState.status = 'ready'
              }
            }
          })
          child.stderr.on('data', (chunk: Buffer) => {
            const lines = chunk.toString().split('\n')
            for (const line of lines) {
              appendLog(line)
            }
          })
          child.on('exit', (code) => {
            const exitCode = code ?? 0
            if (backendState.status !== 'ready') {
              backendState.status = 'error'
              backendState.error = `api.py exited with code ${exitCode}`
            } else {
              backendState.status = 'idle'
            }
            backendState.process = null
          })

          res.statusCode = 200
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ status: backendState.status, apiUrl: backendState.apiUrl }))
        } catch (error) {
          backendState.status = 'error'
          backendState.error = error instanceof Error ? error.message : String(error)
          res.statusCode = 500
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ error: backendState.error }))
        }
      })

      server.middlewares.use('/__backend/status', (req, res, next) => {
        if (req.method !== 'GET') {
          next()
          return
        }
        res.statusCode = 200
        res.setHeader('Content-Type', 'application/json')
        res.end(
          JSON.stringify({
            status: backendState.status,
            apiUrl: backendState.apiUrl,
            error: backendState.error,
            logs: backendState.logs.slice(-20),
          }),
        )
      })
    },
  }
}

export default defineConfig({
  plugins: [
    backendLauncherPlugin(),
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      // Alias @ to the src directory
      '@': path.resolve(__dirname, './src'),
    },
  },

  // File types to support raw imports. Never add .css, .tsx, or .ts files to this.
  assetsInclude: ['**/*.svg', '**/*.csv'],
})
