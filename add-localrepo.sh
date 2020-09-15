#!/bin/sh
lein uberjar
lein pom
lein localrepo install target/clj-djl-0.1.0-SNAPSHOT.jar clj-djl 0.1.0
mv pom.xml ~/.m2/repository/clj-djl/clj-djl/0.1.0/clj-djl-0.1.0.pom
