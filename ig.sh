#!/bin/bash
root_gitignore=".gitignore"
> "$root_gitignore"  # 清空原有的根目录 .gitignore

# 遍历所有子目录的 .gitignore
find . -mindepth 2 -name ".gitignore" | while read file; do
    dir=$(dirname "$file")
    echo "# From $file" >> "$root_gitignore"
    while IFS= read -r line; do
        if [[ -z "$line" || "$line" =~ ^# ]]; then
            # 空行或注释原样写入
            echo "$line" >> "$root_gitignore"
        else
            # 给规则加上子目录前缀
            echo "${dir#./}/$line" >> "$root_gitignore"
        fi
    done < "$file"
    echo "" >> "$root_gitignore"
done

echo "Doc/" >> .gitignore
echo ".history/" >> .gitignore
