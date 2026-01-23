#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Start Script for RLGQAI
快速启动脚本 - 一键安装和运行演示
"""

import subprocess
import sys
import os

def print_banner():
    """打印欢迎横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              RLGQAI - 量子AI自动调优系统                    ║
    ║     Reinforcement Learning for Generative Quantum AI      ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """检查Python版本"""
    print("🔍 检查Python版本...")
    version = sys.version_info
    print(f"   当前版本: Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ 错误: 需要Python 3.7或更高版本")
        return False
    
    print("✅ Python版本符合要求")
    return True

def check_dependencies():
    """检查依赖是否已安装"""
    print("\n🔍 检查依赖包...")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'qiskit': 'Qiskit'
    }
    
    missing = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} (未安装)")
            missing.append(package)
    
    return missing

def install_dependencies():
    """安装依赖"""
    print("\n📦 安装依赖包...")
    print("   这可能需要几分钟时间...\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✅ 依赖安装完成")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ 依赖安装失败")
        return False

def run_demo():
    """运行演示"""
    print("\n🚀 运行演示程序...\n")
    print("=" * 60)
    
    try:
        subprocess.check_call([sys.executable, "demo.py"])
        print("\n" + "=" * 60)
        print("✅ 演示完成")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ 演示运行失败")
        return False

def show_next_steps():
    """显示后续步骤"""
    print("\n" + "=" * 60)
    print("🎉 恭喜！RLGQAI已准备就绪")
    print("=" * 60)
    print("\n📚 后续步骤:\n")
    print("1️⃣  查看完整文档:")
    print("   cat README.md\n")
    print("2️⃣  阅读使用指南:")
    print("   cat USAGE.md\n")
    print("3️⃣  开始训练:")
    print("   python train.py\n")
    print("4️⃣  自定义训练:")
    print("   python train.py --episodes 1000 --batch-size 256\n")
    print("5️⃣  查看所有参数:")
    print("   python train.py --help\n")
    print("=" * 60)
    print("\n💡 提示: 训练过程中的模型将保存在 ./checkpoints/ 目录")
    print("📊 日志文件将保存在 ./logs/ 目录\n")

def main():
    """主函数"""
    print_banner()
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查依赖
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠️  发现 {len(missing)} 个缺失的依赖包")
        response = input("\n是否自动安装? (y/n): ").strip().lower()
        
        if response in ['y', 'yes', '是']:
            if not install_dependencies():
                print("\n❌ 安装失败，请手动运行: pip install -r requirements.txt")
                sys.exit(1)
        else:
            print("\n请手动安装依赖:")
            print("   pip install -r requirements.txt")
            sys.exit(0)
    else:
        print("\n✅ 所有依赖已就绪")
    
    # 询问是否运行演示
    print("\n" + "=" * 60)
    response = input("是否运行快速演示? (y/n): ").strip().lower()
    
    if response in ['y', 'yes', '是']:
        if not run_demo():
            sys.exit(1)
    else:
        print("\n跳过演示")
    
    # 显示后续步骤
    show_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)

