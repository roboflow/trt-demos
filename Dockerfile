FROM nginx
COPY ./conf/roboflow-nginx.conf /etc/nginx/nginx.conf
CMD ["nginx", "-g", "daemon off;"]