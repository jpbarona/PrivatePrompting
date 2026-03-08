import { useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { motion, AnimatePresence } from "motion/react";
import { CircularProgress } from "./components/CircularProgress";
import { Chat } from "./components/Chat";

type BackendStatus = "idle" | "starting" | "ready" | "error";

type BackendStatusResponse = {
  status: BackendStatus;
  apiUrl: string | null;
  error: string | null;
  logs: string[];
};

const IS_PROD = import.meta.env.PROD;
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (IS_PROD ? "/api" : "http://localhost:8000");

export default function App() {
  const [activeTab, setActiveTab] = useState("connect");
  const [showHostProgress, setShowHostProgress] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isHosting, setIsHosting] = useState(false);
  const [peerNumber, setPeerNumber] = useState<number | null>(null);
  const [bootstrapMaddr, setBootstrapMaddr] = useState("");
  const [runId, setRunId] = useState("");
  const [connectStatus, setConnectStatus] = useState<BackendStatus>("idle");
  const [connectError, setConnectError] = useState("");
  const [connectLogs, setConnectLogs] = useState<string[]>([]);
  const [inferUrl, setInferUrl] = useState(`${API_BASE_URL}/infer`);

  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  const pollUntilReady = async () => {
    for (let i = 0; i < 180; i++) {
      const statusRes = await fetch("/__backend/status");
      if (!statusRes.ok) {
        throw new Error(`Status check failed with ${statusRes.status}`);
      }
      const status = (await statusRes.json()) as BackendStatusResponse;
      setConnectStatus(status.status);
      setConnectLogs(status.logs ?? []);
      if (status.status === "ready" && status.apiUrl) {
        setInferUrl(`${status.apiUrl}/infer`);
        setIsConnected(true);
        return;
      }
      if (status.status === "error") {
        throw new Error(status.error ?? "Backend failed to start");
      }
      await sleep(1000);
    }
    throw new Error("Timed out waiting for backend startup");
  };

  const handleConnectClick = async () => {
    setConnectError("");
    setConnectLogs([]);

    if (IS_PROD) {
      setInferUrl(`${API_BASE_URL}/infer`);
      setConnectStatus("ready");
      setIsConnected(true);
      return;
    }

    if (!bootstrapMaddr.trim()) {
      setConnectError("bootstrap_maddr is required");
      return;
    }
    setConnectStatus("starting");
    try {
      const res = await fetch("/__backend/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bootstrapMaddr: bootstrapMaddr.trim(),
          runId: runId.trim(),
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(err.error ?? res.statusText);
      }
      await pollUntilReady();
    } catch (error) {
      setConnectStatus("error");
      setConnectError(error instanceof Error ? error.message : String(error));
    }
  };

  const handleHostClick = () => {
    setShowHostProgress(true);
  };

  const handleHostComplete = () => {
    setTimeout(() => {
      setShowHostProgress(false);
      setIsHosting(true);
      // Generate random number between 100 and 10000
      const randomPeer = Math.floor(Math.random() * (10000 - 100 + 1)) + 100;
      setPeerNumber(randomPeer);
    }, 1000);
  };

  return (
    <div className="size-full bg-black text-white overflow-hidden flex flex-col relative">
      {/* Animated radial gradient background */}
      <motion.div
        className="absolute inset-0 opacity-60"
        animate={{
          background: [
            'radial-gradient(circle at 20% 50%, rgba(59, 130, 246, 0.15) 0%, transparent 50%), radial-gradient(circle at 80% 50%, rgba(139, 92, 246, 0.15) 0%, transparent 50%)',
            'radial-gradient(circle at 80% 30%, rgba(59, 130, 246, 0.15) 0%, transparent 50%), radial-gradient(circle at 20% 70%, rgba(139, 92, 246, 0.15) 0%, transparent 50%)',
            'radial-gradient(circle at 50% 80%, rgba(59, 130, 246, 0.15) 0%, transparent 50%), radial-gradient(circle at 50% 20%, rgba(139, 92, 246, 0.15) 0%, transparent 50%)',
            'radial-gradient(circle at 20% 50%, rgba(59, 130, 246, 0.15) 0%, transparent 50%), radial-gradient(circle at 80% 50%, rgba(139, 92, 246, 0.15) 0%, transparent 50%)',
          ],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "linear",
        }}
      />

      {/* Stars */}
      <div className="absolute inset-0">
        {Array.from({ length: 100 }).map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-0.5 h-0.5 bg-white rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              opacity: [0.2, 1, 0.2],
              scale: [1, 1.5, 1],
            }}
            transition={{
              duration: 2 + Math.random() * 3,
              repeat: Infinity,
              delay: Math.random() * 5,
            }}
          />
        ))}
      </div>

      {/* Content */}
      <div className="relative z-10 flex flex-col size-full">
        {/* Header */}
        <div className="px-8 py-6">
          <h1 className="text-2xl font-light tracking-tight">PrivatePrompt</h1>
        </div>

        {/* Tabs */}
        <Tabs.Root value={activeTab} onValueChange={setActiveTab} className="flex flex-col flex-1 min-h-0">
          <div className="flex justify-center mb-8">
            <Tabs.List className="inline-flex gap-1 p-1 bg-white/5 backdrop-blur-xl rounded-full border border-white/10">
              <Tabs.Trigger
                value="connect"
                className="px-8 py-3 rounded-full text-sm font-medium transition-all duration-300 data-[state=active]:bg-white data-[state=active]:text-slate-950 data-[state=inactive]:text-white/60 data-[state=inactive]:hover:text-white"
              >
                Connect
              </Tabs.Trigger>
              <Tabs.Trigger
                value="host"
                className="px-8 py-3 rounded-full text-sm font-medium transition-all duration-300 data-[state=active]:bg-white data-[state=active]:text-slate-950 data-[state=inactive]:text-white/60 data-[state=inactive]:hover:text-white"
              >
                Host
              </Tabs.Trigger>
            </Tabs.List>
          </div>

          {/* Connect Tab Content */}
          <Tabs.Content value="connect" className="flex-1 min-h-0 flex flex-col">
            {isConnected ? (
              <Chat inferUrl={inferUrl} />
            ) : (
              <div className="flex-1 flex items-center justify-center px-6">
                <div className="w-full max-w-2xl bg-white/5 backdrop-blur-xl rounded-3xl border border-white/10 p-6 space-y-4">
                  <h2 className="text-xl font-light tracking-wide">Launch Backend</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <input
                      value={bootstrapMaddr}
                      onChange={(e) => setBootstrapMaddr(e.target.value)}
                      placeholder="bootstrap_maddr (required)"
                      className="bg-white/5 border border-white/10 rounded-xl px-4 py-3 outline-none md:col-span-2"
                    />
                    <input
                      value={runId}
                      onChange={(e) => setRunId(e.target.value)}
                      placeholder="run_id (optional)"
                      className="bg-white/5 border border-white/10 rounded-xl px-4 py-3 outline-none"
                    />
                  </div>
                  <motion.button
                    onClick={() => void handleConnectClick()}
                    disabled={connectStatus === "starting"}
                    className="w-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl px-4 py-3 font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                  >
                    {connectStatus === "starting" ? "Starting backend..." : "Connect"}
                  </motion.button>
                  {connectStatus === "starting" && (
                    <div className="text-sm text-white/70">Waiting for api.py to become ready.</div>
                  )}
                  {connectStatus === "ready" && (
                    <div className="text-sm text-emerald-300">Connected to {inferUrl}</div>
                  )}
                  {connectError && <div className="text-sm text-red-300">{connectError}</div>}
                  {connectLogs.length > 0 && (
                    <div className="max-h-36 overflow-y-auto text-xs bg-black/40 border border-white/10 rounded-xl p-3 space-y-1">
                      {connectLogs.map((line, idx) => (
                        <div key={`${idx}-${line.slice(0, 24)}`} className="text-white/70">
                          {line}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </Tabs.Content>

          {/* Host Tab Content */}
          <Tabs.Content value="host" className="flex-1 flex items-center justify-center">
            {isHosting ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="text-center"
              >
                <div className="inline-block px-12 py-6 bg-white/5 backdrop-blur-xl rounded-3xl border border-white/10">
                  <p className="text-4xl font-light tracking-wide">
                    Connected as Peer <span className="font-medium bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">{peerNumber}</span>
                  </p>
                </div>
              </motion.div>
            ) : (
              <AnimatePresence mode="wait">
                {!showHostProgress ? (
                  <motion.button
                    key="host-button"
                    onClick={handleHostClick}
                    className="group relative w-48 h-48 bg-white/5 backdrop-blur-xl rounded-full border border-white/10 hover:border-white/30 flex items-center justify-center transition-all duration-300"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    transition={{ duration: 0.3 }}
                  >
                    {/* Subtle gradient overlay on hover */}
                    <motion.div
                      className="absolute inset-0 rounded-full bg-gradient-to-r from-emerald-500/20 via-teal-500/20 to-cyan-500/20 opacity-0 group-hover:opacity-100"
                      transition={{ duration: 0.3 }}
                    />
                    
                    {/* Subtle glow effect */}
                    <div className="absolute inset-0 blur-2xl bg-gradient-to-r from-emerald-500/20 via-teal-500/20 to-cyan-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                    
                    <span className="relative text-xl font-light tracking-wide">
                      Host
                    </span>
                  </motion.button>
                ) : (
                  <motion.div
                    key="host-progress"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    transition={{ duration: 0.3 }}
                  >
                    <CircularProgress onComplete={handleHostComplete} />
                  </motion.div>
                )}
              </AnimatePresence>
            )}
          </Tabs.Content>
        </Tabs.Root>
      </div>
    </div>
  );
}