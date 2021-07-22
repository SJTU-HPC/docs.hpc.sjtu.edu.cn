clean:
	rm -rf /lustre/home/acct-hpc/hpcsjw/test/app/git_demo/docs.hpc.sjtu.edu.cn/_build
	rm -rf /lustre/home/acct-hpc/hpcsjw/test/app/git_demo/docs.hpc.sjtu.edu.cn/reframe-tests-master*
	rm -rf /lustre/home/acct-hpc/hpcsjw/test/app/git_demo/docs.hpc.sjtu.edu.cn/_static/sjtuhpc-checks

%:
	make -C /lustre/home/acct-hpc/hpcsjw/test/app/git_demo/docs.hpc.sjtu.edu.cn $@
