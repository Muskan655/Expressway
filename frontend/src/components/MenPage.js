import React from "react";
import Card from "./Card";
import men_shoes_1 from "../assets/men_shoes/men_shoes_1.png";
import men_shoes_2 from "../assets/men_shoes/men_shoes_2.png";
import men_shoes_3 from "../assets/men_shoes/men_shoes_3.png";
import men_shoes_21 from "../assets/men_shoes/men_shoes_21.png";
import men_shoes_22 from "../assets/men_shoes/men_shoes_22.png";
import men_shoes_23 from "../assets/men_shoes/men_shoes_23.png";
import men_jacket_11 from "../assets/men_jackets/men_jacket_11.png";
import men_jacket_12 from "../assets/men_jackets/men_jacket_12.png";
import men_jacket_13 from "../assets/men_jackets/men_jacket_13.png";
import men_jacket_21 from "../assets/men_jackets/men_jacket_21.png";
import men_jacket_22 from "../assets/men_jackets/men_jacket_22.png";
import men_jacket_23 from "../assets/men_jackets/men_jacket_23.png";
import men_shirt_11 from "../assets/men_shirts/men_shirt_11.png";
import men_shirt_12 from "../assets/men_shirts/men_shirt_12.png";
import men_shirt_13 from "../assets/men_shirts/men_shirt_13.png";
import men_shirt_21 from "../assets/men_shirts/men_shirt_21.png";
import men_shirt_22 from "../assets/men_shirts/men_shirt_22.png";
import men_shirt_23 from "../assets/men_shirts/men_shirt_23.png";
import "./MenPage.css"

const MenPage = () => {
  const menShoes1 = [men_shoes_1, men_shoes_2, men_shoes_3];
  const menShoes2 = [men_shoes_21, men_shoes_22, men_shoes_23];
  const menJackets1 = [men_jacket_11, men_jacket_12, men_jacket_13];
  const menJackets2 = [men_jacket_21, men_jacket_22, men_jacket_23];
  const menShirts1 = [men_shirt_11, men_shirt_12, men_shirt_13];
  const menShirts2 = [men_shirt_21, men_shirt_22, men_shirt_23];
  return (
    <div>
      <div className="men-page-container">
        <div className="card-holder">
          <Card
            images={menJackets1}
            name="Allen Solly"
            description="Men Brown Solid Bomber Jacket"
            price="1484"
            originalPrice="3299"
          />
        </div>
        <div className="card-holder">
          <Card
            images={menJackets2}
            name="Decathlon"
            description="DOMYOS - Men Lightweight Sporty Jacket"
            price="1249"
            originalPrice="1299"
          />
        </div>
        <div className="card-holder">
          <Card
            images={menShirts1}
            name="VASTRADO"
            description="Men Classic Slim Fit Textured Cutaway Collar Cotton Casual Shirt"
            price="1649"
            originalPrice="1649"
          />
        </div>
        <div className="card-holder">
          <Card
            images={menShirts2}
            name="Roadster"
            description="Men Maroon & Grey Slim Fit Checked Casual Shirt"
            price="558"
            originalPrice="1299"
          />
        </div>
      </div>
      <div className="men-page-container">
        <div className="card-holder">
          <Card
            images={menShoes1}
            name="HRX by Hrithik Roshan"
            description="Unisex Back To School Shoes"
            price="999"
            originalPrice="3699"
          />
        </div>
        <div className="card-holder">
          <Card
            images={menShoes2}
            name="Allen solly"
            description="Leather Shoes"
            price="3999"
            originalPrice="5950"
          />
        </div>
      </div>
    </div>
  );
};

export default MenPage;
