# Git Practice 

This github repository is created solely for me to become famliliar with git. 

## Getting Started

Git is a distributed version-control system for tracking changes in source code during software development. It is designed for coordinating work among programmers, but it can be used to track changes in any set of files. Its goals include speed, data integrity, and support for distributed, non-linear workflows (https://en.wikipedia.org/wiki/Git).

### Prerequisites

Working PC (linux OS preferred) such as Ubuntu OS PC. For other platforms, please consult other sources.

### Installing

Git is already installed on many linux distribution (simple terminal command is given below). For other platforms, please google search on your own. 
```
sudo apt install git-all
```

If you are on Fedora (or any closely-related RPM-based distribution, such as RHEL or CentOS), you can use dnf
```
sudo dnf install git-all
```

## Git repository tests 

Create git repository in the current directory:
```
git init
```

Within this directory you can start verson controlling with many git commands.

### Remote repository 

Create Github repository to which to push your git changes. Instructions on how to associate your git directory with Github repository will be provided. For convenient push, on Github repository site, go to deploy keys setting and set up SSH connection (If you already have public and private keys stored in your PC, use that. If you need public and private keys for SSH, use keygen program to generate them).

### Git commands 

Here are frequently used basic git commands. 
```
git add
git commit
git push
```

Git add command updates the index using the current content found in the working tree, to prepare the content staged for the next commit. Git commit records changes to the repository. git push updates remote refs along with associated objects (https://git-scm.com/docs).

## Built With

* [framework](https://www.github.com/sejin8642/gitpractice) - The web framework used (for this repository, there are none)

## Contributing

Please read [CONTRIBUTING.md](.github/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/sejin8642/gitpractice/tags). 

## Authors

* [Devil Man](https://github.com/ackma3141)
* [Sejin Nam](https://github.com/sejin8642)

See also the list of [contributors](https://github.com/sejin8642/gitpractice/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

