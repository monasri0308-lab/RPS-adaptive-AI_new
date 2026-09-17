"""
Native Desktop GUI for Rock-Paper-Scissors Opponent Modeling
=============================================================
A pure-Python Tkinter desktop window for playing live against the Markov AI.
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rps_agent import AdaptiveAgent, COUNTER_MOVES, MOVE_NAMES, evaluate_round


class RPSDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RPS Markov Opponent Modeling AI")
        self.root.geometry("620x680")
        self.root.configure(bg="#0f172a")
        self.root.resizable(False, False)

        self.agent = AdaptiveAgent(order=2)
        self.human_wins = 0
        self.agent_wins = 0
        self.ties = 0
        self.total_rounds = 0

        self._build_ui()

    def _build_ui(self):
        # Header Frame
        header = tk.Frame(self.root, bg="#0f172a", pady=15)
        header.pack(fill="x")

        title = tk.Label(
            header,
            text="Rock-Paper-Scissors AI Arena",
            font=("Helvetica", 18, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
        )
        title.pack()

        subtitle = tk.Label(
            header,
            text="Order-2 Markov Chain Agent with Backoff",
            font=("Helvetica", 10),
            fg="#94a3b8",
            bg="#0f172a",
        )
        subtitle.pack(pady=2)

        # Scoreboard Frame
        score_frame = tk.Frame(self.root, bg="#1e293b", padx=15, pady=12, relief="ridge", bd=1)
        score_frame.pack(fill="x", padx=25, pady=10)

        # 4 columns: Rounds, AI Wins, You, Win Rate
        self.lbl_rounds = tk.Label(score_frame, text="Rounds\n0", font=("Helvetica", 12, "bold"), fg="#f8fafc", bg="#1e293b")
        self.lbl_rounds.grid(row=0, column=0, padx=18)

        self.lbl_ai = tk.Label(score_frame, text="AI Wins\n0", font=("Helvetica", 12, "bold"), fg="#ef4444", bg="#1e293b")
        self.lbl_ai.grid(row=0, column=1, padx=18)

        self.lbl_you = tk.Label(score_frame, text="Your Wins\n0", font=("Helvetica", 12, "bold"), fg="#22c55e", bg="#1e293b")
        self.lbl_you.grid(row=0, column=2, padx=18)

        self.lbl_rate = tk.Label(score_frame, text="AI Win%\n0.0%", font=("Helvetica", 12, "bold"), fg="#38bdf8", bg="#1e293b")
        self.lbl_rate.grid(row=0, column=3, padx=18)

        # Arena Display Box
        arena_frame = tk.Frame(self.root, bg="#1e293b", padx=20, pady=15, relief="ridge", bd=1)
        arena_frame.pack(fill="x", padx=25, pady=10)

        self.lbl_player_move = tk.Label(arena_frame, text="YOU\n❓", font=("Helvetica", 14, "bold"), fg="#f8fafc", bg="#1e293b")
        self.lbl_player_move.grid(row=0, column=0, padx=40)

        vs_lbl = tk.Label(arena_frame, text="VS", font=("Helvetica", 16, "bold"), fg="#64748b", bg="#1e293b")
        vs_lbl.grid(row=0, column=1, padx=20)

        self.lbl_ai_move = tk.Label(arena_frame, text="MARKOV AI\n🤖", font=("Helvetica", 14, "bold"), fg="#f8fafc", bg="#1e293b")
        self.lbl_ai_move.grid(row=0, column=2, padx=40)

        # Outcome Banner
        self.lbl_outcome = tk.Label(
            self.root,
            text="Choose your move (Rock, Paper, or Scissors)",
            font=("Helvetica", 11, "bold"),
            fg="#f8fafc",
            bg="#334155",
            pady=8,
        )
        self.lbl_outcome.pack(fill="x", padx=25, pady=8)

        # Buttons Frame
        btn_frame = tk.Frame(self.root, bg="#0f172a")
        btn_frame.pack(pady=10)

        btn_rock = tk.Button(
            btn_frame,
            text="🪨 Rock",
            font=("Helvetica", 12, "bold"),
            bg="#3b82f6",
            fg="white",
            width=12,
            height=2,
            cursor="hand2",
            command=lambda: self.play("R"),
        )
        btn_rock.grid(row=0, column=0, padx=8)

        btn_paper = tk.Button(
            btn_frame,
            text="📄 Paper",
            font=("Helvetica", 12, "bold"),
            bg="#8b5cf6",
            fg="white",
            width=12,
            height=2,
            cursor="hand2",
            command=lambda: self.play("P"),
        )
        btn_paper.grid(row=0, column=1, padx=8)

        btn_scissors = tk.Button(
            btn_frame,
            text="✂️ Scissors",
            font=("Helvetica", 12, "bold"),
            bg="#ec4899",
            fg="white",
            width=12,
            height=2,
            cursor="hand2",
            command=lambda: self.play("S"),
        )
        btn_scissors.grid(row=0, column=2, padx=8)

        # AI Prediction Insight
        self.lbl_prediction = tk.Label(
            self.root,
            text="AI Inference: Waiting for initial moves...",
            font=("Consolas", 9),
            fg="#94a3b8",
            bg="#0f172a",
        )
        self.lbl_prediction.pack(pady=4)

        # History Treeview
        hist_frame = tk.Frame(self.root, bg="#0f172a")
        hist_frame.pack(fill="both", expand=True, padx=25, pady=10)

        columns = ("round", "you", "ai", "pred", "result")
        self.tree = ttk.Treeview(hist_frame, columns=columns, show="headings", height=5)
        self.tree.heading("round", text="Round")
        self.tree.heading("you", text="Your Move")
        self.tree.heading("ai", text="AI Move")
        self.tree.heading("pred", text="AI Predicted")
        self.tree.heading("result", text="Result")

        self.tree.column("round", width=60, anchor="center")
        self.tree.column("you", width=100, anchor="center")
        self.tree.column("ai", width=100, anchor="center")
        self.tree.column("pred", width=110, anchor="center")
        self.tree.column("result", width=100, anchor="center")
        self.tree.pack(fill="both", expand=True)

        # Reset button
        btn_reset = tk.Button(
            self.root,
            text="Reset Match",
            font=("Helvetica", 9),
            bg="#475569",
            fg="white",
            command=self.reset_game,
        )
        btn_reset.pack(pady=8)

        # Key bindings
        self.root.bind("<r>", lambda e: self.play("R"))
        self.root.bind("<p>", lambda e: self.play("P"))
        self.root.bind("<s>", lambda e: self.play("S"))
        self.root.bind("<R>", lambda e: self.play("R"))
        self.root.bind("<P>", lambda e: self.play("P"))
        self.root.bind("<S>", lambda e: self.play("S"))

    def play(self, move: str):
        emojis = {"R": "🪨", "P": "📄", "S": "✂️"}
        predicted_move = self.agent.model.predict_next_move()
        agent_move = COUNTER_MOVES[predicted_move]
        outcome = evaluate_round(agent_move, move)

        self.agent.observe(move)
        self.total_rounds += 1

        if outcome == "WIN":
            self.agent_wins += 1
            res_str = "AI WINS"
            self.lbl_outcome.config(text=f"AI Won! AI predicted {MOVE_NAMES[predicted_move]} and played {MOVE_NAMES[agent_move]}", bg="#7f1d1d", fg="#fca5a5")
        elif outcome == "LOSS":
            self.human_wins += 1
            res_str = "YOU WIN"
            self.lbl_outcome.config(text=f"You Won! AI played {MOVE_NAMES[agent_move]}", bg="#14532d", fg="#86efac")
        else:
            self.ties += 1
            res_str = "TIE"
            self.lbl_outcome.config(text=f"Tie! Both played {MOVE_NAMES[move]}", bg="#78350f", fg="#fde68a")

        ai_rate = (self.agent_wins / self.total_rounds) * 100

        # Update labels
        self.lbl_rounds.config(text=f"Rounds\n{self.total_rounds}")
        self.lbl_ai.config(text=f"AI Wins\n{self.agent_wins}")
        self.lbl_you.config(text=f"Your Wins\n{self.human_wins}")
        self.lbl_rate.config(text=f"AI Win%\n{ai_rate:.1f}%")

        self.lbl_player_move.config(text=f"YOU\n{emojis[move]} {MOVE_NAMES[move]}")
        self.lbl_ai_move.config(text=f"MARKOV AI\n{emojis[agent_move]} {MOVE_NAMES[agent_move]}")
        self.lbl_prediction.config(text=f"AI Inference: Predicted {MOVE_NAMES[predicted_move]} ({predicted_move}) ➔ Countered with {MOVE_NAMES[agent_move]} ({agent_move})")

        # Insert to history treeview at top
        self.tree.insert("", 0, values=(self.total_rounds, f"{emojis[move]} {move}", f"{emojis[agent_move]} {agent_move}", f"{emojis[predicted_move]} {predicted_move}", res_str))

    def reset_game(self):
        self.agent.reset()
        self.human_wins = 0
        self.agent_wins = 0
        self.ties = 0
        self.total_rounds = 0

        self.lbl_rounds.config(text="Rounds\n0")
        self.lbl_ai.config(text="AI Wins\n0")
        self.lbl_you.config(text="Your Wins\n0")
        self.lbl_rate.config(text="AI Win%\n0.0%")
        self.lbl_player_move.config(text="YOU\n❓")
        self.lbl_ai_move.config(text="MARKOV AI\n🤖")
        self.lbl_outcome.config(text="Game reset. Choose your move!", bg="#334155", fg="#f8fafc")
        self.lbl_prediction.config(text="AI Inference: Model reset to initial state.")
        for item in self.tree.get_children():
            self.tree.delete(item)


if __name__ == "__main__":
    root = tk.Tk()
    app = RPSDesktopApp(root)
    root.mainloop()
