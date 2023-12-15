set -e;
mkdir lra_data;
cd lra_data;
curl https://storage.googleapis.com/long-range-arena/lra_release.gz \
    --output lra_release.tar.gz;
tar xvf lra_release.tar.gz;
mv lra_release/lra_release/listops-1000 listops;
mv lra_release/lra_release/tsv_data matching;
mkdir pathfinder;
mv lra_release/lra_release/pathfinder* pathfinder/;
rm -rf lra_release;
cd ..;
