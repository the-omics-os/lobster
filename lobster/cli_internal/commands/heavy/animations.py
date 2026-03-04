"""
Terminal DNA helix and agent loading animations.

Self-contained animation functions for the Lobster CLI startup,
agent loading, and exit sequences. Extracted from cli.py.
"""

import math
import random
import shutil
import sys
import time
from typing import List

from lobster.ui.console_manager import get_console_manager

console_manager = get_console_manager()
console = console_manager.console


def _dna_helix_animation(width: int, duration: float = 0.7):
    """
    DNA sequence animation with colorful bases (A, T, C, G) flowing across the terminal.
    Uses only capital DNA letters with biochemistry-inspired colors.
    """
    # DNA base colors (biochemistry-inspired)
    base_colors = {
        "A": (0, 200, 83),  # Adenine - green (purine)
        "T": (255, 82, 82),  # Thymine - red (pyrimidine)
        "G": (255, 193, 7),  # Guanine - gold (purine)
        "C": (41, 121, 255),  # Cytosine - blue (pyrimidine)
    }

    lobster_orange = (228, 92, 71)
    white = (255, 255, 255)

    num_frames = max(40, int(duration * 90))
    frame_sleep = duration / num_frames

    # Generate a random DNA sequence (only capital letters A, T, C, G)
    sequence = "".join(random.choices(["A", "T", "G", "C"], k=width))

    # Phase 1: DNA sequence assembly (flowing bases)
    assembly_frames = int(num_frames * 0.45)
    for frame in range(assembly_frames):
        progress = (frame + 1) / assembly_frames
        reveal_pos = int(width * progress)

        line_parts = []
        for i in range(width):
            if i < reveal_pos:
                base = sequence[i]
                r, g, b = base_colors[base]
                # Add glow effect for recently revealed bases
                if i > reveal_pos - 5:
                    glow = 1.3 - (reveal_pos - i) * 0.06
                    r = min(255, int(r * glow))
                    g = min(255, int(g * glow))
                    b = min(255, int(b * glow))
                line_parts.append(f"\033[38;2;{r};{g};{b};1m{base}\033[0m")
            else:
                line_parts.append(" ")

        sys.stdout.write("\r" + "".join(line_parts))
        sys.stdout.flush()
        time.sleep(frame_sleep * 0.5)

    # Phase 2: DNA wave intensity effect (only capital letters)
    helix_frames = int(num_frames * 0.35)
    for frame in range(helix_frames):
        line_parts = []
        wave_offset = frame * 0.4

        for i in range(width):
            base = sequence[i]
            # Sinusoidal wave for intensity variation (simulates depth)
            wave = math.sin((i * 0.25) + wave_offset)
            intensity = 0.5 + 0.5 * abs(wave)

            r, g, b = base_colors[base]
            r = int(r * intensity)
            g = int(g * intensity)
            b = int(b * intensity)

            line_parts.append(f"\033[38;2;{r};{g};{b};1m{base}\033[0m")

        sys.stdout.write("\r" + "".join(line_parts))
        sys.stdout.flush()
        time.sleep(frame_sleep * 0.8)

    # Phase 3: Orange sweep with centered "-- omics-os --" brand text
    sweep_frames = num_frames - assembly_frames - helix_frames
    brand_text = "\u2500 omics-os \u2500"
    text_start = (width - len(brand_text)) // 2
    text_end = text_start + len(brand_text)

    for frame in range(sweep_frames + 1):
        progress = frame / sweep_frames if sweep_frames > 0 else 1
        sweep_pos = int(width * progress)

        line_parts = []
        for i in range(width):
            if i < sweep_pos:
                # Check if this position is part of the brand text
                if text_start <= i < text_end:
                    char_idx = i - text_start
                    char = brand_text[char_idx]
                    # White text on orange background
                    r, g, b = lobster_orange
                    line_parts.append(
                        f"\033[48;2;{r};{g};{b}m\033[38;2;255;255;255;1m{char}\033[0m"
                    )
                else:
                    # Gradient orange fill
                    gradient = 1.0 - (sweep_pos - i) / max(width, 1) * 0.12
                    r = min(255, int(lobster_orange[0] * gradient))
                    g = min(255, int(lobster_orange[1] * gradient))
                    b = min(255, int(lobster_orange[2] * gradient))
                    line_parts.append(f"\033[48;2;{r};{g};{b}m \033[0m")
            elif i == sweep_pos:
                # Bright leading edge
                line_parts.append(
                    f"\033[48;2;{white[0]};{white[1]};{white[2]}m \033[0m"
                )
            else:
                # Fading DNA bases behind
                base = sequence[i]
                fade = max(0.15, 1.0 - (i - sweep_pos) * 0.08)
                r, g, b = base_colors[base]
                r, g, b = int(r * fade), int(g * fade), int(b * fade)
                line_parts.append(f"\033[38;2;{r};{g};{b};1m{base}\033[0m")

        sys.stdout.write("\r" + "".join(line_parts))
        sys.stdout.flush()
        time.sleep(frame_sleep * 1.0)

    # Final orange bar with centered brand text
    r, g, b = lobster_orange
    final_parts = []
    for i in range(width):
        if text_start <= i < text_end:
            char = brand_text[i - text_start]
            final_parts.append(
                f"\033[48;2;{r};{g};{b}m\033[38;2;255;255;255;1m{char}\033[0m"
            )
        else:
            final_parts.append(f"\033[48;2;{r};{g};{b}m \033[0m")
    sys.stdout.write("\r" + "".join(final_parts) + "\n")
    sys.stdout.flush()


def _dna_agent_loading_phase(
    width: int, agent_names: List[str], ready_queue=None, timeout: float = 10.0
):
    """
    DNA-themed agent loading animation showing real-time progress.

    Displays a progress bar made of DNA bases that fills as agents load,
    with each agent name appearing with a DNA sequencing effect.
    """
    import queue as queue_module

    # DNA base colors (biochemistry-inspired)
    base_colors = {
        "A": (0, 200, 83),  # Adenine - green
        "T": (255, 82, 82),  # Thymine - red
        "G": (255, 193, 7),  # Guanine - gold
        "C": (41, 121, 255),  # Cytosine - blue
    }
    total_agents = len(agent_names)
    loaded_count = 0
    start_time = time.time()

    # Generate DNA sequence for progress bar
    sequence = "".join(random.choices(["A", "T", "G", "C"], k=width))

    def render_progress_bar(progress: float, agent_name: str = "", wave_frame: int = 0):
        """Render the DNA progress bar with current progress."""
        filled_width = int(width * progress)

        line_parts = []
        for i in range(width):
            base = sequence[i]
            r, g, b = base_colors[base]

            if i < filled_width:
                # Filled section: bright with wave effect
                wave = math.sin((i * 0.3) + (wave_frame * 0.3))
                intensity = 0.8 + 0.2 * abs(wave)
                r = min(255, int(r * intensity))
                g = min(255, int(g * intensity))
                b = min(255, int(b * intensity))
                line_parts.append(f"\033[38;2;{r};{g};{b};1m{base}\033[0m")
            elif i == filled_width and progress < 1.0:
                # Leading edge: white glow
                line_parts.append(f"\033[38;2;255;255;255;1m{base}\033[0m")
            else:
                # Unfilled section: dim
                r, g, b = int(r * 0.2), int(g * 0.2), int(b * 0.2)
                line_parts.append(f"\033[38;2;{r};{g};{b};1m{base}\033[0m")

        # Clear line and render
        sys.stdout.write("\r" + "".join(line_parts))
        sys.stdout.flush()

        # Show agent name below progress bar (if provided)
        if agent_name:
            # Create compact DNA spinner
            spinner_bases = ["A", "T", "G", "C"]
            spinner_idx = wave_frame % len(spinner_bases)
            spinner_base = spinner_bases[spinner_idx]
            r, g, b = base_colors[spinner_base]

            # Format agent name with DNA spinner
            agent_text = f"  \033[38;2;{r};{g};{b};1m{spinner_base}\033[0m \033[38;2;200;200;200m{agent_name}\033[0m"

            # Move cursor down, print agent name, move cursor back up
            sys.stdout.write(f"\n{agent_text}\033[K")  # \033[K clears to end of line
            sys.stdout.write("\033[1A")  # Move cursor up 1 line
            sys.stdout.flush()

    # Initial render: empty progress bar
    render_progress_bar(0.0)
    sys.stdout.write("\n\n")  # Reserve space for agent name
    sys.stdout.write("\033[2A")  # Move cursor back to progress bar line
    sys.stdout.flush()

    # Animation loop
    wave_frame = 0
    last_agent_name = ""

    if ready_queue:
        # Real-time mode: wait for agent notifications
        while loaded_count < total_agents:
            # Check for timeout
            if time.time() - start_time > timeout:
                break

            try:
                # Non-blocking check for new agent
                agent_loaded = ready_queue.get(timeout=0.05)
                if agent_loaded == "__done__":
                    break
                loaded_count += 1
                last_agent_name = agent_loaded

                # Animate progress increase
                old_progress = (loaded_count - 1) / total_agents
                new_progress = loaded_count / total_agents

                # Smooth transition over 10 frames
                for frame in range(10):
                    progress = old_progress + (new_progress - old_progress) * (
                        frame / 10
                    )
                    render_progress_bar(progress, last_agent_name, wave_frame)
                    wave_frame += 1
                    time.sleep(0.03)

            except queue_module.Empty:
                # Keep animating while waiting
                progress = loaded_count / total_agents
                render_progress_bar(progress, last_agent_name, wave_frame)
                wave_frame += 1
                time.sleep(0.05)
    else:
        # Fallback mode: simulate loading (no real progress)
        estimated_time_per_agent = timeout / total_agents

        for agent_idx, agent_name in enumerate(agent_names):
            agent_start = time.time()

            # Animate this agent loading
            while time.time() - agent_start < estimated_time_per_agent:
                progress = (
                    agent_idx + (time.time() - agent_start) / estimated_time_per_agent
                ) / total_agents
                progress = min(progress, 1.0)
                render_progress_bar(progress, agent_name, wave_frame)
                wave_frame += 1
                time.sleep(0.05)

            loaded_count += 1

    # Final render: complete bar
    render_progress_bar(1.0, "ready", wave_frame)

    # Hold final state briefly
    for _ in range(10):
        render_progress_bar(1.0, "ready", wave_frame)
        wave_frame += 1
        time.sleep(0.04)

    # Clear the agent name line and progress bar
    sys.stdout.write("\r" + " " * width + "\n")  # Clear progress bar
    sys.stdout.write(" " * width + "\n")  # Clear agent name
    sys.stdout.write("\033[2A\r")  # Move cursor back up and to start
    sys.stdout.flush()


def _dna_exit_animation(width: int, duration: float = 0.5):
    """
    DNA exit animation - reverse of startup animation.
    Orange bar dissolves back into DNA bases, which then fade out.
    The brand text ALWAYS stays lobster orange.
    """
    # DNA base colors (biochemistry-inspired)
    base_colors = {
        "A": (0, 200, 83),  # Adenine - green (purine)
        "T": (255, 82, 82),  # Thymine - red (pyrimidine)
        "G": (255, 193, 7),  # Guanine - gold (purine)
        "C": (41, 121, 255),  # Cytosine - blue (pyrimidine)
    }

    lobster_orange = (228, 92, 71)
    white = (255, 255, 255)

    num_frames = max(30, int(duration * 70))
    frame_sleep = duration / num_frames

    # Generate a random DNA sequence
    sequence = "".join(random.choices(["A", "T", "G", "C"], k=width))

    brand_text = "\u2500 omics-os \u2500"
    text_start = (width - len(brand_text)) // 2
    text_end = text_start + len(brand_text)

    # Phase 1: Start with full orange bar, then reverse sweep (right to left dissolve)
    dissolve_frames = int(num_frames * 0.5)
    for frame in range(dissolve_frames + 1):
        progress = frame / dissolve_frames if dissolve_frames > 0 else 1
        dissolve_pos = width - int(width * progress)

        line_parts = []
        for i in range(width):
            if text_start <= i < text_end:
                char = brand_text[i - text_start]
                r, g, b = lobster_orange
                line_parts.append(f"\033[38;2;{r};{g};{b};1m{char}\033[0m")
            elif i >= dissolve_pos:
                base = sequence[i]
                r, g, b = base_colors[base]
                if i < dissolve_pos + 5:
                    glow = 1.3 - (i - dissolve_pos) * 0.06
                    r = min(255, int(r * glow))
                    g = min(255, int(g * glow))
                    b = min(255, int(b * glow))
                line_parts.append(f"\033[38;2;{r};{g};{b};1m{base}\033[0m")
            elif i == dissolve_pos - 1 and dissolve_pos > 0:
                line_parts.append(
                    f"\033[48;2;{white[0]};{white[1]};{white[2]}m \033[0m"
                )
            else:
                fade = max(0.4, 1.0 - progress * 0.6)
                r, g, b = lobster_orange
                r, g, b = int(r * fade), int(g * fade), int(b * fade)
                line_parts.append(f"\033[48;2;{r};{g};{b}m \033[0m")

        sys.stdout.write("\r" + "".join(line_parts))
        sys.stdout.flush()
        time.sleep(frame_sleep * 0.8)

    # Phase 2: DNA wave effect with brand text staying orange
    wave_frames = int(num_frames * 0.25)
    for frame in range(wave_frames):
        line_parts = []
        wave_offset = frame * 0.5

        for i in range(width):
            if text_start <= i < text_end:
                char = brand_text[i - text_start]
                r, g, b = lobster_orange
                line_parts.append(f"\033[38;2;{r};{g};{b};1m{char}\033[0m")
            else:
                base = sequence[i]
                wave = math.sin((i * 0.25) + wave_offset)
                intensity = 0.5 + 0.5 * abs(wave)

                r, g, b = base_colors[base]
                r = int(r * intensity)
                g = int(g * intensity)
                b = int(b * intensity)

                line_parts.append(f"\033[38;2;{r};{g};{b};1m{base}\033[0m")

        sys.stdout.write("\r" + "".join(line_parts))
        sys.stdout.flush()
        time.sleep(frame_sleep * 0.6)

    # Phase 3: Fade out DNA sequence, brand text stays orange then fades last
    fade_frames = num_frames - dissolve_frames - wave_frames
    for frame in range(fade_frames + 1):
        progress = frame / fade_frames if fade_frames > 0 else 1
        fade_intensity = 1.0 - progress
        brand_fade = max(0, 1.0 - max(0, (progress - 0.7) / 0.3))

        line_parts = []
        for i in range(width):
            if text_start <= i < text_end:
                char = brand_text[i - text_start]
                r, g, b = lobster_orange
                if brand_fade > 0.05:
                    r = int(r * brand_fade)
                    g = int(g * brand_fade)
                    b = int(b * brand_fade)
                    line_parts.append(f"\033[38;2;{r};{g};{b};1m{char}\033[0m")
                else:
                    line_parts.append(" ")
            elif fade_intensity > 0.05:
                base = sequence[i]
                r, g, b = base_colors[base]
                r = int(r * fade_intensity)
                g = int(g * fade_intensity)
                b = int(b * fade_intensity)
                line_parts.append(f"\033[38;2;{r};{g};{b};1m{base}\033[0m")
            else:
                line_parts.append(" ")

        sys.stdout.write("\r" + "".join(line_parts))
        sys.stdout.flush()
        time.sleep(frame_sleep * 1.0)

    # Clear the line
    sys.stdout.write("\r" + " " * width + "\r")
    sys.stdout.flush()


def display_welcome():
    """Display DNA sequence animation as bioinformatics-themed startup visualization."""
    term_width = shutil.get_terminal_size().columns
    sys.stdout.write("\n")
    _dna_helix_animation(term_width, duration=random.uniform(0.6, 0.8))
    sys.stdout.write("\n")


def display_goodbye():
    """Display DNA exit animation as bioinformatics-themed farewell visualization."""
    term_width = shutil.get_terminal_size().columns
    sys.stdout.write("\n")
    _dna_exit_animation(term_width, duration=random.uniform(0.4, 0.6))
