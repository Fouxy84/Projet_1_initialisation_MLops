FROM fluent/fluentd:v1.16

USER root

RUN gem uninstall fluent-plugin-elasticsearch -a -x || true
RUN gem install fluent-plugin-elasticsearch -v 7.4.3 --no-document

USER fluent