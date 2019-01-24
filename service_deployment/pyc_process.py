import os, shutil
import sys
import compileall

_FILEPATH = os.path.dirname(os.path.abspath(__file__))

# 需要保存的py文件
_needsavepyfile = ['pyc_process.py', 'main.py']
# cpy文件名中需要去除的部分
_needdelstr = 'cpython-36.'

# 生产py文件的pyc文件, 给绝对路径
COMPILE_DIR = sys.argv[1]
compileall.compile_dir(COMPILE_DIR)


def code_cpy(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件

        current_path = os.path.join(file_dir, root)
        for file in files:
            if file in _needsavepyfile:
                continue

            if str(file).endswith('.py'):
                os.remove(os.path.join(current_path, file))
                print('delete {}'.format(file))

        for dir in dirs:
            if str(dir) == '__pycache__':
                pyc_path = os.path.join(current_path, '__pycache__')
                for file in os.listdir(pyc_path):
                    if str(file).endswith('.pyc'):
                        file_path = os.path.join(pyc_path, file)
                        shutil.move(file_path, current_path)  # 文件复制
                        new_name = str(file).replace(_needdelstr, '') # cpy文件名处理
                        new_file = os.path.join(current_path, new_name)
                        os.rename(os.path.join(current_path, file), new_file)  # 文件重命名

                # shutil.rmtree(pyc_path) # 删除cpy文件夹


code_cpy(COMPILE_DIR)
