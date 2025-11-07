# debug_resume.py
import sys
import os

def debug_resume_param():
    """调试恢复参数传递"""
    print("=== Debug Resume Parameter ===")
    
    # 检查命令行参数
    print("Command line arguments:", sys.argv)
    
    # 检查是否有 --resume 参数
    if '--resume' in sys.argv:
        resume_index = sys.argv.index('--resume')
        if resume_index + 1 < len(sys.argv):
            resume_path = sys.argv[resume_index + 1]
            print(f"Found --resume parameter: {resume_path}")
            
            # 检查文件是否存在
            if os.path.exists(resume_path):
                print(f"✓ Checkpoint file exists: {resume_path}")
                print(f"File size: {os.path.getsize(resume_path)} bytes")
            else:
                print(f"✗ Checkpoint file not found: {resume_path}")
        else:
            print("✗ --resume parameter provided but no path specified")
    else:
        print("✗ No --resume parameter found in command line")

if __name__ == "__main__":
    debug_resume_param()