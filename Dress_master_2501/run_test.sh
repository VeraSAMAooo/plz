list_root="/home/user/development/datasets/bottom_lower_crop/demo"
skirt_type_path="${list_root}/skirt_types.txt"
IFS="
"
skirt_types=$(cat $skirt_type_path)
for skirt_type in $skirt_types
do 
    echo "${skirt_type}..."
    pass_list="${list_root}/${skirt_type}.txt"
    output_path="./test_outputs/${skirt_type}"
    tocg_model="./checkpoint/v4/tocg_skirt/skirt_99000.pth"
    python test.py --pass_list $pass_list --output_root $output_path --tocg_model $tocg_model
done 

trouser_type_path="${list_root}/trouser_types.txt"
trouser_types=$(cat $trouser_type_path)
for trouser_type in $trouser_types
do 
    echo "${trouser_type}..."
    pass_list="${list_root}/${trouser_type}.txt"
    output_path="./test_outputs/${trouser_type}"
    tocg_model="./checkpoint/v4/tocg_trousers/trousers_99000.pth"
    python test.py --pass_list $pass_list --output_root $output_path --tocg_model $tocg_model
done 

echo "done"