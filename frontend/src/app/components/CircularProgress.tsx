import { motion } from "motion/react";
import { useEffect, useState } from "react";

interface CircularProgressProps {
  onComplete?: () => void;
}

export function CircularProgress({ onComplete }: CircularProgressProps) {
  const [progress, setProgress] = useState(0);
  const radius = 60;
  const strokeWidth = 6;
  const circumference = 2 * Math.PI * radius;

  useEffect(() => {
    const steps = [33.33, 66.66, 100];
    let currentStep = 0;

    const interval = setInterval(() => {
      if (currentStep < steps.length) {
        setProgress(steps[currentStep]);
        currentStep++;
      } else {
        clearInterval(interval);
        if (onComplete) {
          setTimeout(onComplete, 300);
        }
      }
    }, 600);

    return () => clearInterval(interval);
  }, [onComplete]);

  const getDotPosition = (percentage: number) => {
    // Calculate angle starting from top (12 o'clock position)
    // Since SVG is rotated -90deg, we don't need to subtract 90 here
    const angle = (percentage / 100) * 360;
    const radian = (angle * Math.PI) / 180;
    const x = 80 + radius * Math.cos(radian);
    const y = 80 + radius * Math.sin(radian);
    return { x, y };
  };

  const dots = [
    { percentage: 33.33, active: progress >= 33.33 },
    { percentage: 66.66, active: progress >= 66.66 },
    { percentage: 100, active: progress >= 100 },
  ];

  return (
    <div className="relative w-40 h-40 flex items-center justify-center">
      <svg
        width="160"
        height="160"
        className="transform -rotate-90"
      >
        {/* Background circle */}
        <circle
          cx="80"
          cy="80"
          r={radius}
          stroke="rgba(255, 255, 255, 0.1)"
          strokeWidth={strokeWidth}
          fill="none"
        />
        
        {/* Animated progress circle */}
        <motion.circle
          cx="80"
          cy="80"
          r={radius}
          stroke="url(#gradient)"
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{
            strokeDashoffset: circumference - (progress / 100) * circumference,
          }}
          transition={{
            duration: 0.5,
            ease: [0.4, 0, 0.2, 1],
          }}
        />
        
        {/* Gradient definition */}
        <defs>
          <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#10b981" />
            <stop offset="50%" stopColor="#34d399" />
            <stop offset="100%" stopColor="#6ee7b7" />
          </linearGradient>
        </defs>
        
        {/* Dots at 1/3 intervals */}
        {dots.map((dot, index) => {
          const pos = getDotPosition(dot.percentage);
          return (
            <motion.circle
              key={index}
              cx={pos.x}
              cy={pos.y}
              r={dot.active ? 5 : 3}
              fill={dot.active ? "#fff" : "rgba(255, 255, 255, 0.3)"}
              initial={{ scale: 0, opacity: 0 }}
              animate={{
                scale: dot.active ? [1, 1.3, 1] : 1,
                opacity: dot.active ? 1 : 0.5,
              }}
              transition={{
                duration: 0.3,
                ease: "easeOut",
              }}
            />
          );
        })}
      </svg>
      
      {/* Center text */}
      <motion.div
        className="absolute inset-0 flex items-center justify-center"
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2 }}
      >
        <span className="text-sm text-white/70">
          {progress === 100 ? "Complete" : `${Math.round(progress)}%`}
        </span>
      </motion.div>
    </div>
  );
}