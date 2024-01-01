
echo "************************************************************" | tee -a acc_asr.txt
echo "test the accuracy and attack success rate of the fruit model" | tee -a acc_asr.txt
echo "************************************************************" | tee -a acc_asr.txt

python tflite2pytorch.py --model_name=fruit --save_onnx
python attack.py  --adv=BIM --model=fruit --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=fruit --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=fruit --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=fruit --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=fruit --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=fruit --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt


echo "************************************************************" | tee -a acc_asr.txt
echo "test the accuracy and attack success rate of the skin model" | tee -a acc_asr.txt
echo "************************************************************" | tee -a acc_asr.txt

python tflite2pytorch.py --model_name=skin --save_onnx
python attack.py  --adv=BIM --model=skin --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=skin --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=skin --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=skin --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=skin --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=skin --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt


echo "************************************************************" | tee -a acc_asr.txt
echo "test the accuracy and attack success rate of the imagenet model" | tee -a acc_asr.txt
echo "************************************************************" | tee -a acc_asr.txt

python tflite2pytorch.py --model_name=imagenet --save_onnx
python attack.py  --adv=BIM --model=imagenet --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=imagenet --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=imagenet --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=imagenet --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=imagenet --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=imagenet --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt


echo "************************************************************" | tee -a acc_asr.txt
echo "test the accuracy and attack success rate of the american_sign_language model" | tee -a acc_asr.txt
echo "************************************************************" | tee -a acc_asr.txt

python tflite2pytorch.py --model_name=american_sign_language --save_onnx
python attack.py  --adv=BIM --model=american_sign_language --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=american_sign_language --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=american_sign_language --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=american_sign_language --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=american_sign_language --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=american_sign_language --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt


echo "************************************************************" | tee -a acc_asr.txt
echo "test the accuracy and attack success rate of the plant model" | tee -a acc_asr.txt
echo "************************************************************" | tee -a acc_asr.txt

python tflite2pytorch.py --model_name=plant --save_onnx
python attack.py  --adv=BIM --model=plant --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=plant --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=plant --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=plant --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=plant --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=plant --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt


echo "************************************************************" | tee -a acc_asr.txt
echo "test the accuracy and attack success rate of the cassava model" | tee -a acc_asr.txt
echo "************************************************************" | tee -a acc_asr.txt

python tflite2pytorch.py --model_name=cassava --save_onnx
python attack.py  --adv=BIM --model=cassava --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=cassava --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=cassava --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=cassava --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=cassava --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=cassava --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt


echo "************************************************************" | tee -a acc_asr.txt
echo "test the accuracy and attack success rate of the plant_disease model" | tee -a acc_asr.txt
echo "************************************************************" | tee -a acc_asr.txt

python tflite2pytorch.py --model_name=plant_disease --save_onnx
python attack.py  --adv=BIM --model=plant_disease --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=plant_disease --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=plant_disease --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=plant_disease --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=plant_disease --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=plant_disease --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt


echo "************************************************************" | tee -a acc_asr.txt
echo "test the accuracy and attack success rate of the insect model" | tee -a acc_asr.txt
echo "************************************************************" | tee -a acc_asr.txt

python tflite2pytorch.py --model_name=insect --save_onnx
python attack.py  --adv=BIM --model=insect --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=insect --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=insect --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=insect --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=insect --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=insect --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt


echo "************************************************************" | tee -a acc_asr.txt
echo "test the accuracy and attack success rate of the bird model" | tee -a acc_asr.txt
echo "************************************************************" | tee -a acc_asr.txt

python tflite2pytorch.py --model_name=bird --save_onnx
python attack.py  --adv=BIM --model=bird --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=bird --eps=0.01 --nb_iter=400 --eps_iter=0.0001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=bird --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=bird --eps=0.1 --nb_iter=400 --eps_iter=0.001 | tee -a acc_asr.txt
python attack.py  --adv=BIM --model=bird --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt
python attack.py  --adv=PGD --model=bird --eps=1.0 --nb_iter=400 --eps_iter=0.04 | tee -a acc_asr.txt




