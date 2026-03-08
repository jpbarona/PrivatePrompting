import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Send } from "lucide-react";

const DEFAULT_INFER_URL = "http://localhost:8000/infer";

interface Message {
  id: string;
  text: string;
  sender: "user" | "host";
  timestamp: Date;
}

interface ChatProps {
  inferUrl?: string;
}

export function Chat({ inferUrl = DEFAULT_INFER_URL }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const isNearBottom = () => {
    const el = scrollRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  const scrollToBottom = (force = false) => {
    if (force || isNearBottom()) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: "user",
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    const sentPrompt = inputValue;
    setInputValue("");
    setIsLoading(true);
    scrollToBottom(true); // always scroll when user sends

    try {
      const res = await fetch(inferUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: sentPrompt }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? res.statusText);
      }
      const data = await res.json() as { response: string };
      const hostMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: data.response,
        sender: "host",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, hostMessage]);
    } catch (e) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: `Error: ${e instanceof Error ? e.message : String(e)}`,
        sender: "host",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full w-full max-w-4xl mx-auto px-6 py-8">
      {/* Chat Header */}
      <motion.div
        className="mb-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h2 className="text-xl font-light">Connected</h2>
        <p className="text-sm text-white/50">Secure chat session</p>
      </motion.div>

      {/* Messages Container */}
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto mb-6 space-y-4 pr-2 chat-scroll"
      >
        <AnimatePresence initial={false}>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[70%] px-5 py-3 rounded-2xl ${
                  message.sender === "user"
                    ? "bg-gradient-to-r from-emerald-500 to-teal-500 text-white"
                    : "bg-white/10 backdrop-blur-sm text-white border border-white/10"
                }`}
              >
                <p className="text-sm leading-relaxed">{message.text}</p>
                <span className="text-xs opacity-60 mt-1 block">
                  {message.timestamp.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </span>
              </div>
            </motion.div>
          ))}
          {/* Loading bubble */}
          {isLoading && (
            <motion.div
              key="loading"
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className="flex justify-start"
            >
              <div className="px-5 py-3 rounded-2xl bg-white/10 backdrop-blur-sm text-white border border-white/10">
                <span className="flex gap-1 items-center h-5">
                  {[0, 0.2, 0.4].map((delay) => (
                    <motion.span
                      key={delay}
                      className="w-1.5 h-1.5 rounded-full bg-white/60"
                      animate={{ opacity: [0.3, 1, 0.3], y: [0, -4, 0] }}
                      transition={{ duration: 0.8, repeat: Infinity, delay }}
                    />
                  ))}
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <motion.div
        className="relative"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="flex gap-3 items-end bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-3">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type a message..."
            className="flex-1 bg-transparent border-none outline-none text-white placeholder:text-white/40 px-2 py-2"
          />
          <motion.button
            onClick={() => void handleSend()}
            disabled={!inputValue.trim() || isLoading}
            className="bg-gradient-to-r from-emerald-500 to-teal-500 p-3 rounded-xl disabled:opacity-40 disabled:cursor-not-allowed"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Send className="size-5" />
          </motion.button>
        </div>
      </motion.div>
    </div>
  );
}
