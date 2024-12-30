
    document.addEventListener("DOMContentLoaded", function() {
        fetch('api_data.json')
            .then(response => response.json())
            .then(products => {
                const productContainer = document.getElementById('products');
                productContainer.innerHTML = '';  // 清空当前内容

                products.forEach(product => {
                    const productDiv = document.createElement('div');
                    productDiv.className = 'product';

                    const img = document.createElement('img');
                    img.src = product.image;
                    img.alt = product.name;

                    const name = document.createElement('p');
                    name.textContent = product.name;

                    const price = document.createElement('p');
                    price.textContent = `$${product.price}`;

                    productDiv.appendChild(img);
                    productDiv.appendChild(name);
                    productDiv.appendChild(price);
                    productContainer.appendChild(productDiv);
                });

                document.getElementById('loading').style.display = 'none';
            })
            .catch(error => {
                console.error('Error fetching products:', error);
            });
    });
    