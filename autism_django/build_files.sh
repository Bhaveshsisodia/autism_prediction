
echo "BUILD START"
set -e

python3.12 -m ensurepip --upgrade
python3.12 -m pip install --upgrade pip
python3.12 -m pip install requirements.txt
python3.12 manage.py collectstatic --noinput --clear
echo "BUILD END"