
echo "BUILD START"
set -e

python3.12 -m ensurepip --upgrade
python3.12 -m pip install --upgrade pip
pip install --no-cache-dir -r /vercel/path0/autism_django/requirements.txt
python3.12 manage.py collectstatic --noinput --clear
echo "BUILD END"