mkdir -p data

echo "download official data..."
git clone https://www.modelscope.cn/datasets/issaccv/aiops2024-challenge-dataset.git data
cd data
mkdir format_data_with_img
cp ../src/data/imgmap_filtered.json format_data_with_img/
mkdir origin_data

echo "Unzip files..."
unzip -q -O gb2312 director.zedx -d origin_data/director
unzip -q -O utf-8 emsplus.zedx -d origin_data/emsplus
unzip -q -O gb2312 rcp.zedx -d origin_data/rcp
unzip -q -O utf-8 umac.zedx -d origin_data/umac
mv origin_data/umac/documents/Namf_MP/zh-CN origin_data/umac/documents/Namf_MP/zh-cn
echo "Unzip Done"
cd ..

echo "Preprocess files..."
cd src
python preprocess_zedx.py
cd ..
