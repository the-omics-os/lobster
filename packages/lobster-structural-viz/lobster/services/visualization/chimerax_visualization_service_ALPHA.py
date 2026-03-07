"""
ChimeraX Visualization Service for creating protein structure visualizations.

This stateless service provides methods for generating high-quality 3D protein
structure visualizations using the ChimeraX Python API, following the Lobster
3-tuple pattern (Dict, stats, IR).
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ChimeraXVisualizationError(Exception):
    """Base exception for ChimeraX visualization operations."""

    pass


class ChimeraXVisualizationService:
    """
    Stateless service for creating protein structure visualizations with ChimeraX.

    This service implements the Lobster 3-tuple pattern:
    - Returns (Dict, stats_dict, AnalysisStep)
    - Generates ChimeraX command scripts
    - Creates high-quality structure images
    - Handles ChimeraX installation checks
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the ChimeraX visualization service.

        Args:
            config: Optional configuration dict (for future use)
            **kwargs: Additional arguments (ignored, for backward compatibility)
        """
        logger.debug("Initializing stateless ChimeraXVisualizationService")
        self.config = config or {}
        self._chimerax_available = None
        logger.debug("ChimeraXVisualizationService initialized successfully")

    def visualize_structure(
        self,
        structure_file: Path,
        mode: str = "interactive",
        style: str = "cartoon",
        color_by: str = "chain",
        output_image: Optional[Path] = None,
        width: int = 1920,
        height: int = 1080,
        background: str = "white",
        execute_commands: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Create 3D visualization of protein structure using ChimeraX.

        Args:
            structure_file: Path to PDB/CIF structure file
            mode: Execution mode - 'interactive' (launch GUI) or 'batch' (save image and exit)
            style: Representation style ('cartoon', 'surface', 'sticks', 'spheres')
            color_by: Coloring scheme ('chain', 'secondary_structure', 'bfactor', 'hydrophobicity')
            output_image: Path for output image (PNG), auto-generated if None
            width: Image width in pixels
            height: Image height in pixels
            background: Background color ('white', 'black', 'transparent')
            execute_commands: Whether to execute ChimeraX commands (requires installation)

        Returns:
            Tuple[Dict, Dict, AnalysisStep]: Visualization metadata, stats, and IR

        Raises:
            ChimeraXVisualizationError: If visualization fails
        """
        try:
            logger.info(
                f"Creating ChimeraX visualization: {structure_file} (style: {style}, color: {color_by})"
            )

            # Validate structure file
            structure_file = Path(structure_file)
            if not structure_file.exists():
                raise ChimeraXVisualizationError(
                    f"Structure file not found: {structure_file}"
                )

            # Set output image path
            if output_image is None:
                output_dir = structure_file.parent / "visualizations"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_image = (
                    output_dir / f"{structure_file.stem}_{style}_{color_by}.png"
                )
            output_image = Path(output_image)
            output_image.parent.mkdir(parents=True, exist_ok=True)

            # Generate ChimeraX command script
            commands = self._generate_chimerax_commands(
                structure_file=structure_file,
                mode=mode,
                style=style,
                color_by=color_by,
                output_image=output_image,
                width=width,
                height=height,
                background=background,
            )

            # Save command script
            script_file = output_image.parent / f"{output_image.stem}_commands.cxc"
            with open(script_file, "w") as f:
                f.write("\n".join(commands))

            logger.info(f"ChimeraX command script saved: {script_file}")

            # Execute ChimeraX commands if requested
            executed = False
            execution_message = "Script generated (execution skipped)"
            process_info = {}

            if execute_commands:
                chimerax_installed = self.check_chimerax_installation()
                if chimerax_installed["installed"]:
                    try:
                        if mode == "interactive":
                            # Launch ChimeraX GUI in interactive mode (non-blocking)
                            result = self._launch_chimerax_interactive(
                                script_file, chimerax_installed["path"]
                            )
                            executed = True
                            execution_message = result["message"]
                            process_info = {"pid": result["pid"], "mode": "interactive"}
                            logger.info(f"ChimeraX GUI launched: {result['message']}")
                        else:
                            # Batch mode: execute and save image (blocking)
                            self._execute_chimerax_batch(
                                script_file, chimerax_installed["path"]
                            )
                            executed = True
                            execution_message = (
                                "Successfully executed with ChimeraX (batch mode)"
                            )
                            logger.info(
                                f"ChimeraX visualization created: {output_image}"
                            )
                    except Exception as e:
                        logger.error(f"ChimeraX execution failed: {e}")
                        execution_message = f"Execution failed: {e}"
                else:
                    execution_message = (
                        f"ChimeraX not installed: {chimerax_installed['message']}"
                    )
                    logger.warning(execution_message)

            # Prepare visualization data
            visualization_data = {
                "structure_file": str(structure_file),
                "output_image": str(output_image),
                "script_file": str(script_file),
                "mode": mode,
                "style": style,
                "color_by": color_by,
                "width": width,
                "height": height,
                "background": background,
                "commands": commands,
                "executed": executed,
                "execution_message": execution_message,
                **process_info,  # Add PID if interactive mode
            }

            # Calculate statistics
            stats = {
                "structure_file": structure_file.name,
                "output_image": output_image.name,
                "mode": mode,
                "style": style,
                "color_by": color_by,
                "image_dimensions": f"{width}x{height}",
                "executed": executed,
                "chimerax_commands": len(commands),
                "analysis_type": "protein_structure_visualization",
            }

            # Check if image was created
            if output_image.exists():
                stats["output_file_size_mb"] = output_image.stat().st_size / (
                    1024 * 1024
                )

            logger.info(f"Visualization complete: {stats}")

            # Create IR for notebook export
            ir = self._create_visualization_ir(
                structure_file=structure_file,
                style=style,
                color_by=color_by,
                width=width,
                height=height,
                background=background,
            )

            return visualization_data, stats, ir

        except Exception as e:
            logger.exception(f"Error creating ChimeraX visualization: {e}")
            raise ChimeraXVisualizationError(
                f"Failed to create visualization: {str(e)}"
            )

    def check_chimerax_installation(self) -> Dict[str, Any]:
        """
        Check if ChimeraX is installed and accessible.

        Returns:
            Dict with installation status, path, and version information
        """
        if self._chimerax_available is not None:
            return self._chimerax_available

        # Common ChimeraX installation paths
        potential_paths = [
            "/Applications/ChimeraX.app/Contents/bin/ChimeraX",  # macOS
            "C:\\Program Files\\ChimeraX\\bin\\ChimeraX.exe",  # Windows
            "/usr/bin/chimerax",  # Linux (system)
            "/usr/local/bin/chimerax",  # Linux (local)
        ]

        # Check if chimerax is in PATH
        try:
            result = subprocess.run(
                ["chimerax", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                self._chimerax_available = {
                    "installed": True,
                    "path": "chimerax",
                    "version": version,
                    "message": f"ChimeraX found in PATH: {version}",
                }
                return self._chimerax_available
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Check common installation paths
        for path in potential_paths:
            if Path(path).exists():
                try:
                    result = subprocess.run(
                        [path, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        self._chimerax_available = {
                            "installed": True,
                            "path": path,
                            "version": version,
                            "message": f"ChimeraX found at {path}: {version}",
                        }
                        return self._chimerax_available
                except Exception:
                    continue

        # ChimeraX not found
        self._chimerax_available = {
            "installed": False,
            "path": None,
            "version": None,
            "message": "ChimeraX not found. Install from https://www.cgl.ucsf.edu/chimerax/download.html",
        }
        return self._chimerax_available

    def _generate_chimerax_commands(
        self,
        structure_file: Path,
        mode: str,
        style: str,
        color_by: str,
        output_image: Path,
        width: int,
        height: int,
        background: str,
    ) -> list:
        """Generate ChimeraX command script for visualization."""
        commands = [
            "# ChimeraX Protein Structure Visualization Script",
            "# Generated by Lobster - Protein Structure Visualization Service",
            "",
            "# Open structure file",
            f"open {structure_file}",
            "",
            "# Set background color",
            f"set bgColor {background}",
            "",
            "# Apply representation style",
        ]

        # Style commands
        style_commands = {
            "cartoon": ["hide atoms", "cartoon"],
            "surface": ["hide atoms", "surface"],
            "sticks": ["hide atoms", "show atoms", "style stick"],
            "spheres": ["hide atoms", "show atoms", "style sphere"],
            "ball_and_stick": ["hide atoms", "show atoms", "style ball"],
        }

        commands.extend(style_commands.get(style, style_commands["cartoon"]))
        commands.append("")

        # Color commands
        commands.append("# Apply coloring scheme")
        if color_by == "chain":
            commands.append("color bychain")
        elif color_by == "secondary_structure":
            # Color by secondary structure elements (no byhetero - that colors heteroatoms)
            commands.append("color helix cornflower blue")
            commands.append("color strand yellow")
            commands.append("color coil white")
        elif color_by == "bfactor":
            commands.append("color bfactor palette rainbow")
        elif color_by == "hydrophobicity":
            # WARNING: bykdhydrophobicity not verified in ChimeraX docs - may need testing
            commands.append("color bykdhydrophobicity")
        else:
            commands.append("color bychain")

        commands.extend(
            [
                "",
                "# Center and orient structure",
                "view",
                "",
                "# Set window size",
                f"windowsize {width} {height}",
            ]
        )

        # Mode-specific final commands
        if mode == "batch":
            commands.extend(
                [
                    "",
                    "# Save image and exit",
                    f"save {output_image} supersample 3",
                    "exit",
                ]
            )
        else:  # interactive mode
            commands.extend(
                [
                    "",
                    "# Interactive mode - ChimeraX GUI will remain open",
                    "# You can now interact with the 3D structure",
                    f"# To save an image: save {output_image} supersample 3",
                ]
            )

        return commands

    def _execute_chimerax_batch(self, script_file: Path, chimerax_path: str):
        """Execute ChimeraX command script in batch mode (headless, blocking)."""
        try:
            result = subprocess.run(
                [chimerax_path, "--nogui", "--script", str(script_file)],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise ChimeraXVisualizationError(
                    f"ChimeraX batch execution failed: {result.stderr}"
                )

            logger.debug(f"ChimeraX batch output: {result.stdout}")

        except subprocess.TimeoutExpired:
            raise ChimeraXVisualizationError("ChimeraX batch execution timed out (60s)")
        except Exception as e:
            raise ChimeraXVisualizationError(
                f"Failed to execute ChimeraX in batch mode: {e}"
            )

    def _launch_chimerax_interactive(
        self, script_file: Path, chimerax_path: str
    ) -> Dict[str, Any]:
        """
        Launch ChimeraX GUI in interactive mode (non-blocking).

        Args:
            script_file: Path to ChimeraX command script
            chimerax_path: Path to ChimeraX executable

        Returns:
            Dict with success status, PID, and message

        Raises:
            ChimeraXVisualizationError: If launch fails
        """
        import time

        try:
            # Launch ChimeraX without --nogui (opens GUI window)
            # Use Popen for non-blocking execution
            process = subprocess.Popen(
                [chimerax_path, "--script", str(script_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Detach from parent process
            )

            # Give ChimeraX time to start
            time.sleep(2)

            # Check if process is still running
            poll_result = process.poll()
            if poll_result is not None:
                # Process terminated immediately - likely an error
                stderr_output = (
                    process.stderr.read().decode("utf-8")
                    if process.stderr
                    else "No error output"
                )
                raise ChimeraXVisualizationError(
                    f"ChimeraX failed to start (exit code {poll_result}): {stderr_output}"
                )

            logger.info(f"ChimeraX GUI launched successfully (PID: {process.pid})")

            return {
                "success": True,
                "pid": process.pid,
                "message": f"ChimeraX GUI launched successfully (PID {process.pid})",
            }

        except Exception as e:
            logger.error(f"Failed to launch ChimeraX interactive mode: {e}")
            raise ChimeraXVisualizationError(f"Failed to launch ChimeraX GUI: {e}")

    def _create_visualization_ir(
        self,
        structure_file: Path,
        style: str,
        color_by: str,
        width: int,
        height: int,
        background: str,
    ) -> AnalysisStep:
        """Create Intermediate Representation for visualization operation."""
        parameter_schema = {
            "structure_file": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=str(structure_file),
                required=True,
                description="Path to protein structure file",
            ),
            "style": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=style,
                required=False,
                validation_rule="style in ['cartoon', 'surface', 'sticks', 'spheres', 'ball_and_stick']",
                description="Representation style",
            ),
            "color_by": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=color_by,
                required=False,
                validation_rule="color_by in ['chain', 'secondary_structure', 'bfactor', 'hydrophobicity']",
                description="Coloring scheme",
            ),
            "width": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=width,
                required=False,
                description="Image width in pixels",
            ),
            "height": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=height,
                required=False,
                description="Image height in pixels",
            ),
        }

        code_template = """# Create protein structure visualization with ChimeraX
import subprocess
from pathlib import Path

structure_file = "{{ structure_file }}"
style = "{{ style }}"
color_by = "{{ color_by }}"
width = {{ width }}
height = {{ height }}

# Generate ChimeraX commands
commands = [
    f"open {structure_file}",
    "set bgColor white",
    "hide atoms",
]

# Style command
if style == "cartoon":
    commands.append("cartoon")
elif style == "surface":
    commands.append("surface")
elif style in ["sticks", "spheres"]:
    commands.extend(["show atoms", f"style {style}"])

# Color commands
if color_by == "chain":
    commands.append("color bychain")
elif color_by == "secondary_structure":
    commands.extend([
        "color helix cornflower blue",
        "color strand yellow",
        "color coil white"
    ])
elif color_by == "bfactor":
    commands.append("color bfactor palette rainbow")
elif color_by == "hydrophobicity":
    commands.append("color bykdhydrophobicity")

# Finalize
commands.extend([
    "view",
    f"windowsize {width} {height}",
    f"save {Path(structure_file).stem}_visualization.png supersample 3",
    "exit"
])

# Save command script
script_file = Path(f"{Path(structure_file).stem}_chimerax_commands.cxc")
script_file.write_text("\\n".join(commands))

print(f"ChimeraX command script created: {script_file}")
print("Run with: chimerax --nogui --script {script_file}")

# Optional: Execute if ChimeraX is installed
try:
    result = subprocess.run(
        ["chimerax", "--nogui", "--script", str(script_file)],
        capture_output=True,
        timeout=60
    )
    if result.returncode == 0:
        print(f"Visualization created successfully")
    else:
        print(f"ChimeraX execution failed: {result.stderr}")
except FileNotFoundError:
    print("ChimeraX not found in PATH. Install from https://www.cgl.ucsf.edu/chimerax/")
"""

        return AnalysisStep(
            operation="chimerax.visualize_structure",
            tool_name="visualize_with_chimerax",
            description=f"Create {style} visualization of protein structure colored by {color_by}",
            library="chimerax",
            code_template=code_template,
            imports=["import subprocess", "from pathlib import Path"],
            parameters={
                "structure_file": str(structure_file),
                "style": style,
                "color_by": color_by,
                "width": width,
                "height": height,
                "background": background,
            },
            parameter_schema=parameter_schema,
            input_entities=[str(structure_file)],
            output_entities=[f"{structure_file.stem}_visualization.png"],
            execution_context={
                "operation_type": "visualization",
                "tool": "ChimeraX",
                "visualization_type": "3d_structure",
            },
            validates_on_export=True,
            requires_validation=False,
        )
