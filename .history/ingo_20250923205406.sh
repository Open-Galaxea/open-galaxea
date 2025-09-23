# 遍历每个子目录的 .gitignore
for dir in G0 GalaxeaDP GalaxeaLeRobot GalaxeaManipSim; do
  if [ -f "$dir/.gitignore" ]; then
    # 给每行规则加上前缀
    sed "s|^|$dir/|" "$dir/.gitignore" >> .gitignore
  fi
done