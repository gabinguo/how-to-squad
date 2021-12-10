#! /bin/bash

# Usage
# ./mlm_train_test_split.sh abc.txt 80 20
# => Generate two files: 80% in train-[filename] and 20% in test-[filename]

file="$1"
fileLength=$(wc -l < "$file")
shift

part=1
percentSum=0
currentLine=1
for percent in "$@"; do
        [ "$percent" == "." ] && ((percent = 100 - percentSum))
        ((percentSum += percent))
        if ((percent < 0 || percentSum > 100)); then
                echo "invalid percentage" 1>&2
                exit 1
        fi
        ((nextLine = fileLength * percentSum / 100))
        if ((nextLine < currentLine)); then
                printf "" # create empty file
        else
                sed -n "$currentLine,$nextLine"p "$file"
        fi > "part$part-$file"
        ((currentLine = nextLine + 1))
        ((part++))
done

mv "part1-$file" "train-$file"
mv "part2-$file" "test-$file"
