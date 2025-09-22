# Test deploy fastapi 
{ pkgs ? import <nixpkgs> {} }:

let
    pythonEnv = pkgs.python3.withPackages (p: with p; [
    fastapi
    uvicorn
    # ... other dependencies from your requirements.txt
    ]);
in
pkgs.stdenv.mkDerivation {
    pname = "my-fastapi-app";
    version = "0.1.0";
    src = ./.; # Assuming your app is in the current directory

    buildInputs = [ pythonEnv ];

    installPhase = ''
    mkdir -p $out/bin
    cp -r . $out/app
    echo "#!${pythonEnv}/bin/python" > $out/bin/run-app
    echo "import uvicorn" >> $out/bin/run-app
    echo "import sys" >> $out/bin/run-app
    echo "sys.path.append('$out/app')" >> $out/bin/run-app
    echo "uvicorn.run('main:app', host='0.0.0.0', port=8080)" >> $out/bin/run-app
    chmod +x $out/bin/run-app
    '';
}