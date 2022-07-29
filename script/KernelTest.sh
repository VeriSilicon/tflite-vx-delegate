#! /bin/bash
# export VIVANTE_SDK_DIR=
# export LD_LIBRARY_PATH=
# export DISABLE_IDE_DEBUG=1
# export VIV_VX_DEBUG_LEVEL=1
# export VSI_NN_LOG_LEVEL=5

OPtotal=0
OPpass=0
OPfail=0
OPcrash=0
file_path=$(pwd)
op_path=`dirname $(pwd)`/build/_deps/tensorflow-build/kernels/
delegate_path=`dirname $(pwd)`/build/libvx_delegate.so
> $file_path/opresult.csv
> $file_path/caseres.txt


### This function is used to get the full name of each case in the given op
function getFull(){
    $1$2 --external_delegate_path=$3 --gtest_list_tests | tee -a $file_path/mylist.txt >/dev/null 2>&1
    > $file_path/caselist.txt
    cat $file_path/mylist.txt | while read rows
    do
     temp=$rows
        if [[ "$temp" == *"."* ]]
        then
            parname=$temp
            # parname=$temp | cut -d"#" -f1
        elif [[ "$temp" != "DISABLED_"* ]]
        then
            fullname=${parname%"."*}"."${temp}
            # fullname=${parname}${temp}
            echo $fullname >> $file_path/caselist.txt
        fi
    done
    rm -f $file_path/mylist.txt
}

function getop(){
    ls -l $1 | grep "_test" | awk '{print $9}' | tee $file_path/oplist.txt >/dev/null 2>&1
}
getop $op_path



while read rows
do
    op_name=$rows
    getFull $op_path $op_name $delegate_path

    clist=$file_path/caselist.txt
    > $file_path/temp.txt
    > $file_path/tempres.txt
    
    cat $clist | cut -d" " -f1 | while read rows
    do
        check_res=`"$op_path""$op_name" --external_delegate_path=$delegate_path --gtest_filter="$rows" | grep -Eom1 "PASSED|FAILED"`
        if [ ! $check_res ]
            then echo "CRASHED" >> $file_path/temp.txt
        else
            echo ${check_res} >> $file_path/temp.txt
        fi
    done


    paste $clist $file_path/temp.txt > $file_path/tempres.txt
    rm -f $file_path/temp.txt
    total=`wc -l $file_path/tempres.txt | awk '{print $1}'`
    pass=`grep -c "PASSED" $file_path/tempres.txt`
    fail=`grep -c "FAILED" $file_path/tempres.txt`
    crash=`grep -c "CRASHED" $file_path/tempres.txt`   
    echo  $op_name $total,$pass,$fail,$crash >> $file_path/opresult.csv
    OPtotal=`expr $OPtotal + 1`
        
    if [ $total -ne $pass ]
    then
    echo "OP $op_name is not full passed:" >> $file_path/caseres.txt
    echo "The Failed cases listed below: " >> $file_path/caseres.txt
    grep "FAILED" $file_path/tempres.txt | awk '{print $1}' >> $file_path/caseres.txt
    echo  "The Crashed cases listed below: " >> $file_path/caseres.txt
    grep "CRASHED" $file_path/tempres.txt| awk '{print $1}' >> $file_path/caseres.txt
    echo "-----------------------------------------------------" >> $file_path/caseres.txt
    fi
        
    if [ $fail -gt 0 ]
    then OPfail=`expr $OPfail + 1`
    elif [ $crash -gt 0 ]
    then OPcrash=`expr $OPcrash + 1`
    elif [ $pass -gt 0 ]
    then OPpass=`expr $OPpass + 1`
    # echo $OPtotal $OPpass $OPfail $OPcrash
    fi

done  <<<"$(cat $file_path/oplist.txt)"

rm -f $file_path/caselist.txt
rm -f $file_path/tempres.txt
echo "-------------Kernel Test Finished------------- "
echo "$OPtotal ops have tested this time, with the result that "
echo "Full passed ops: $OPpass "
echo "Failed ops: $OPfail"
echo "Crashed ops: $OPcrash"
# echo $OPpass > OPres.txt
# echo $OPpass >> $GITHUB_ENV