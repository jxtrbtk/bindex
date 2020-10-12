FROM debian:buster
RUN apt-get update && apt-get install -y python3 python3-pip git && pip3 install requests pandas

CMD token=$(cat /secret/github.txt) && \
git clone --depth 1 https://jxtrbtk:$token@github.com/jxtrbtk/bindex.git && \
python3 bindex/collector.py