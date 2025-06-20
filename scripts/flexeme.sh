#!/bin/sh
# source: https://github.com/PPPI/Flexeme

BASE_DIR="datasets/candidate/ㅡㄴ flexeme/subjects"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR" || exit 1

repos="
Commandline https://github.com/commandlineparser/commandline.git 67f77e1
CommonMark https://github.com/Knagis/CommonMark.NET.git f3d5453
Hangfire https://github.com/HangfireIO/Hangfire.git 175207c
Humanizer https://github.com/Humanizr/Humanizer.git 604ebcc
Lean https://github.com/QuantConnect/Lean.git 71bc0fa
Nancy https://github.com/NancyFx/Nancy.git dbdbe94
NewtonsoftJson https://github.com/JamesNK/Newtonsoft.Json.git 4f8832a
Ninject https://github.com/ninject/ninject.git 6a7ed2b
RestSharp https://github.com/restsharp/RestSharp.git b52b9be
"

echo "🔧 Starting cloning..."

echo "$repos" | while read -r name url sha; do
    if [ -z "$name" ]; then continue; fi

    echo "➡️  Cloning $name..."
    git clone "$url" "$name"
    cd "$name" || continue
    git reset --hard "$sha"
    cd ..
done

echo "✅ All repositories cloned and reset under: $BASE_DIR"
