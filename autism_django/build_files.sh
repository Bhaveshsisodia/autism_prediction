
echo "BUILD START"
set -e
python3 --version
python3.9 --version
python3.9 -m ensurepip --upgrade
python3.9 -m pip install --upgrade pip
pip install --no-cache-dir -r /vercel/path0/autism_django/requirements.txt
python3.9 manage.py collectstatic --noinput --clear
echo "BUILD END"