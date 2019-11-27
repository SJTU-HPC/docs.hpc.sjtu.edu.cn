.PHONY : update server

git:
	git pull
	git push

server:
	mkdocs serve

update:
	rm -rf public
	mkdocs build --clean
	rm -rf static
	mkdir static
	scp -r root@docs.hpc.sjtu.edu.cn:/usr/local/webserver/nginx/html/* static/
	cp -r public/* static/
	scp -r static/* root@docs.hpc.sjtu.edu.cn:/usr/local/webserver/nginx/html/
	rm -rf static
