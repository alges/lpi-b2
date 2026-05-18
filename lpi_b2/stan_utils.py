from cmdstanpy import cmdstan_path, install_cmdstan


def ensure_cmdstan_installed():
    """Install CmdStan if it is not already available."""
    try:
        path = cmdstan_path()
        print(f"CmdStan already installed at: {path}")
    except Exception:
        print("CmdStan not found. Installing now...")
        install_cmdstan(overwrite=True)
        print("CmdStan installation complete.")
