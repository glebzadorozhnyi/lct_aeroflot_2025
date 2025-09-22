    # nix/fastapi-app.nix
    { pkgs, pythonEnv }:

    pkgs.writeShellApplication {
      name = "fastapi-app";
      runtimeInputs = [ pythonEnv ];
      text = ''
        uvicorn my_fastapi_app.main:app --host 0.0.0.0 --port 8000
      '';
    }nox --list