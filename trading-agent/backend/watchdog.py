import sys
import time
import subprocess
import socket

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0

def cleanup():
    processes_to_kill = ["ollama.exe", "redis-server.exe", "node.exe"]
    for proc in processes_to_kill:
        # /T kills child processes as well
        subprocess.call(['taskkill', '/F', '/T', '/IM', proc], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    # Only allow one instance of watchdog at a time
    lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock_socket.bind(('127.0.0.1', 8001))
    except OSError:
        # Another watchdog is already running, exit silently
        sys.exit(0)
    
    # Wait initially for apps to bind ports
    time.sleep(15)
    
    down_count = 0
    while True:
        backend_up = check_port(8000)
        frontend_up = check_port(3000)
        
        # We need to detect if BOTH the frontend and backend are definitely closed/killed.
        if not backend_up and not frontend_up:
            down_count += 5
        elif not backend_up: 
            down_count += 2
        elif backend_up and frontend_up:
            down_count = 0
            
        if down_count >= 15:
            # Double check for false flags
            time.sleep(3)
            if not check_port(8000) and not check_port(3000):
                cleanup()
                break
                
        time.sleep(5)

if __name__ == "__main__":
    main()
