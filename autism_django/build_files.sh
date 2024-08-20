
echo " BUILD START"
python3.9.19 -m pip install -r requirements.txt
python3.9.19 manage.py collectstatic --noinput --clear
echo " BUILD END"