.PHONY : update server

git:
	git pull
	git push

server:
	mkdocs serve

update:
	mkdocs build --clean
	scp -r public/* root@docs.hpc.sjtu.edu.cn:/usr/local/webserver/nginx/html/