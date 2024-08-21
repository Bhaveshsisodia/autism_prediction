
echo "BUILD START"
python3.9 -m pip install django==4.2.15
python3.9 -m pip install -r requirements.txt
python3.9 manage.py collectstatic --noinput --clear
echo "BUILD END"