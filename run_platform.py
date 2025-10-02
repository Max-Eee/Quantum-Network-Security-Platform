#!/usr/bin/env python3
"""
Easy Platform Launcher - Start the Quantum Network Security Platform

This script provides easy access to all platform features.
"""

import os
import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def print_banner():
    print("🚀 " + "="*60)
    print("🌟 QUANTUM NETWORK SECURITY PLATFORM LAUNCHER 🌟")
    print("="*64)
    print()

def run_demo():
    """Run the comprehensive platform demonstration"""
    print("🔬 Running Comprehensive Platform Demo...")
    print("-" * 40)
    try:
        result = subprocess.run([sys.executable, "live_demo.py"], 
                              cwd=Path(__file__).parent, 
                              capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    print("🌐 Starting Platform API Server...")
    print("📡 Server will be available at: http://localhost:8080")
    print("📊 API Documentation at: http://localhost:8080/docs")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Give user a moment to see the info
        time.sleep(2)
        webbrowser.open("http://localhost:8080/docs")
        
        # Change to src directory and run from there
        src_path = Path(__file__).parent / "src"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent)
        
        result = subprocess.run([sys.executable, "main.py", "start", "--port", "8080"], 
                              cwd=src_path,
                              env=env)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def run_simulation():
    """Run quantum network simulation"""
    print("⚛️ Starting Quantum Network Simulation...")
    print("-" * 40)
    try:
        # Change to src directory and run from there
        src_path = Path(__file__).parent / "src"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent)
        
        result = subprocess.run([sys.executable, "main.py", "simulate", 
                               "--nodes", "5", "--duration", "60"], 
                              cwd=src_path,
                              env=env)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        return False

def show_menu():
    """Display the main menu"""
    print("🎯 Choose an option:")
    print("1. 🔬 Run Platform Demo (Full feature showcase)")
    print("2. 🌐 Start API Server (REST API + Documentation)")
    print("3. ⚛️ Run Quantum Simulation (Network topology simulation)")
    print("4. 📊 Show Platform Architecture")
    print("5. ❓ Help & Documentation")
    print("0. 🚪 Exit")
    print()

def show_architecture():
    """Show platform architecture overview"""
    print("📊 " + "="*60)
    print("🏗️  PLATFORM ARCHITECTURE OVERVIEW")
    print("="*64)
    print()
    print("🔬 Quantum Computing Layer:")
    print("   ├── TensorFlow Quantum Integration (fallback mode)")
    print("   ├── Quantum Circuit Management")
    print("   ├── Quantum Machine Learning")
    print("   └── Quantum Algorithms (VQE, QAOA)")
    print()
    print("🔐 Security & Cryptography Layer:")
    print("   ├── Quantum Key Distribution (BB84, E91, SARG04)")
    print("   ├── Eavesdropping Detection")
    print("   ├── Post-Quantum Cryptography")
    print("   └── Security Event Monitoring")
    print()
    print("🧠 AI & Machine Learning Layer:")
    print("   ├── Quantum-Enhanced Anomaly Detection")
    print("   ├── Network Behavior Analysis")
    print("   ├── Predictive Security Monitoring")
    print("   └── Adaptive Threat Response")
    print()
    print("🌐 Network & API Layer:")
    print("   ├── FastAPI REST Interface")
    print("   ├── Real-time WebSocket Updates")
    print("   ├── Interactive Network Visualization")
    print("   └── Performance Monitoring Dashboard")
    print("="*64)
    print()

def show_help():
    """Show help and documentation"""
    print("❓ " + "="*60)
    print("📚 PLATFORM HELP & DOCUMENTATION")
    print("="*64)
    print()
    print("📖 Key Files:")
    print("   • live_demo.py - Comprehensive feature demonstration")
    print("   • src/main.py - Main platform entry point")
    print("   • README.md - Full documentation")
    print("   • PLATFORM_CAPABILITIES.md - Detailed capabilities")
    print()
    print("🌐 API Endpoints (when server is running):")
    print("   • GET /api/network/status - Network status")
    print("   • GET /api/security/threats - Security threats")
    print("   • POST /api/qkd/session - Start QKD session")
    print("   • GET /api/metrics/performance - Performance metrics")
    print("   • WS /api/events - Real-time event stream")
    print()
    print("💡 Pro Tips:")
    print("   • All quantum features work in simulation mode")
    print("   • Platform is production-ready for demonstrations")
    print("   • Use requirements-minimal.txt for Python 3.13")
    print("   • Use requirements.txt for full TensorFlow Quantum (Python 3.8-3.11)")
    print("="*64)
    print()

def main():
    """Main application loop"""
    print_banner()
    
    while True:
        show_menu()
        
        try:
            choice = input("👉 Enter your choice (0-5): ").strip()
            print()
            
            if choice == '1':
                run_demo()
            elif choice == '2':
                start_server()
            elif choice == '3':
                run_simulation()
            elif choice == '4':
                show_architecture()
            elif choice == '5':
                show_help()
            elif choice == '0':
                print("👋 Thank you for using the Quantum Network Security Platform!")
                print("🌟 Keep exploring quantum technologies! 🌟")
                break
            else:
                print("❌ Invalid choice. Please enter a number from 0-5.")
            
            input("\n⏎ Press Enter to continue...")
            print("\n" + "="*64 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            input("\n⏎ Press Enter to continue...")

if __name__ == "__main__":
    main()